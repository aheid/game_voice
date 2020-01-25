const DeepSpeech = require('deepspeech');
const VAD = require('node-vad');
const mic = require('micmod');
const fs = require('fs');
const wav = require('wav');
const Speaker = require('speaker');

let DEEPSPEECH_MODEL; // path to deepspeech model directory
if (process.env.DEEPSPEECH_MODEL) {
	DEEPSPEECH_MODEL = process.env.DEEPSPEECH_MODEL;
}
else {
	DEEPSPEECH_MODEL = __dirname + '/deepspeech-0.6.0-models';
}

let DEEPSPEECH_LM; // path to language model
if (process.env.DEEPSPEECH_LM) {
	DEEPSPEECH_LM = process.env.DEEPSPEECH_LM;
}
else {
	DEEPSPEECH_LM = DEEPSPEECH_MODEL;
}


let SILENCE_THRESHOLD = 200; // how many milliseconds of inactivity before processing the audio

//const VAD_MODE = VAD.Mode.NORMAL;
// const VAD_MODE = VAD.Mode.LOW_BITRATE;
//const VAD_MODE = VAD.Mode.AGGRESSIVE;
const VAD_MODE = VAD.Mode.VERY_AGGRESSIVE;
const vad = new VAD(VAD_MODE);

function createModel(modelDir, lmDir, options) {
	let modelPath = modelDir + '/output_graph.pbmm';
	let lmPath = lmDir + '/lm.binary';
	let triePath = lmDir + '/trie';
	let model = new DeepSpeech.Model(modelPath, options.BEAM_WIDTH);
	model.enableDecoderWithLM(lmPath, triePath, options.LM_ALPHA, options.LM_BETA);
	return model;
}

let englishModel = createModel(
	DEEPSPEECH_MODEL, 
	DEEPSPEECH_LM,
	{
		BEAM_WIDTH: 1024,
		LM_ALPHA: 0.75,
		LM_BETA: 1.85
	}
);

let modelStream;
let recordedChunks = 0;
let silenceStart = null;
let recordedAudioLength = 0;
let endTimeout = null;
let silenceBuffers = [];
let firstChunkVoice = false;
let vadPromise = null;
let dataIdx = 0;
let dataChunks = new Array();
let dataResults = new Map(); // dataIdx => vad result
var outfile = fs.createWriteStream('output.raw');

function processClassifiedData(vadRes, data, callback) {
	
	if (firstChunkVoice) {
		firstChunkVoice = false;
		// ignore first chunk
		//processVoice(data);
		return;
	}
	
	switch (vadRes) {
		case VAD.Event.ERROR:
			console.log("VAD ERROR");
			break;
		case VAD.Event.NOISE:
			console.log("VAD NOISE");
			break;
		case VAD.Event.SILENCE:
			processSilence(data, callback);
			break;
		case VAD.Event.VOICE:
			processVoice(data);
			break;
		default:
			console.log('default', vadRes);
	}
}

function processAudioStream(data, callback) {

	// half-assed attempt at serializing the vad results
	// the issue is that vad.processAudio spawns an
	// AsyncWorker, but we need to feed DeepSpeech the chunks
	// in the right order... 
	// thus, to ensure this, instead maintain a queue of chunks
	// and pop the first one off and feed that to DeepSpeech

	dataChunks.push(
		{
			idx: dataIdx,
			data: data			
		}
	);

	resFn = function(resIdx) {
		return (res) => {
		
			dataResults.set(resIdx, res);
	
			var d = dataChunks.shift();

			if (!dataResults.has(d.idx))
			{
				console.log('missing result: ', d.idx);
			}

			var vadRes = dataResults.get(d.idx);
			dataResults.delete(d.idx);
	
			processClassifiedData(vadRes, d.data, callback);
		};
	};

	vad.processAudio(data, 16000).then(resFn(dataIdx));
	
	dataIdx = dataIdx + 1;

	// timeout after 1s of inactivity
	clearTimeout(endTimeout);
	endTimeout = setTimeout(function() {
		console.log('timeout');
		resetAudioStream();
	},SILENCE_THRESHOLD*3);
}

function endAudioStream(callback) {
	console.log('[end]');
	let results = intermediateDecode();
	if (results) {
		if (callback) {
			callback(results);
		}
	}
}

function resetAudioStream() {
	clearTimeout(endTimeout);
	console.log('[reset]');
	intermediateDecode(); // ignore results
	recordedChunks = 0;
	silenceStart = null;
}

function processSilence(data, callback) {
	if (recordedChunks > 0) { // recording is on
		process.stdout.write('-'); // silence detected while recording
		
		feedAudioContent(data);
		
		if (silenceStart === null) {
			silenceStart = new Date().getTime();
		}
		else {
			let now = new Date().getTime();
			if (now - silenceStart > SILENCE_THRESHOLD) {
				silenceStart = null;
				console.log('[end]');
				let results = intermediateDecode();
				if (results) {
					if (callback) {
						callback(results);
					}
				}
			}
		}
	}
	else {
		process.stdout.write('.'); // silence detected while not recording
		bufferSilence(data);
	}
}

function bufferSilence(data) {
	// VAD has a tendency to cut the first bit of audio data from the start of a recording
	// so keep a buffer of that first bit of audio and in addBufferedSilence() reattach it to the beginning of the recording
	silenceBuffers.push(data);
	if (silenceBuffers.length >= 3) {
		silenceBuffers.shift();
	}
}

function addBufferedSilence(data) {
	let audioBuffer;
	if (silenceBuffers.length) {
		silenceBuffers.push(data);
		let length = 0;
		silenceBuffers.forEach(function (buf) {
			length += buf.length;
		});
		audioBuffer = Buffer.concat(silenceBuffers, length);
		silenceBuffers = [];
	}
	else audioBuffer = data;
	return audioBuffer;
}

function processVoice(data) {
	silenceStart = null;
	if (recordedChunks === 0) {
		console.log('');
		process.stdout.write('[start]'); // recording started
	}
	else {
		process.stdout.write('='); // still recording
	}
	recordedChunks++;
	
	data = addBufferedSilence(data);
	feedAudioContent(data);
}

function createStream() {
	modelStream = englishModel.createStream();
	recordedChunks = 0;
	recordedAudioLength = 0;
}

function finishStream() {
	if (modelStream) {
		let start = new Date();
		let text = englishModel.finishStream(modelStream);
		if (text) {
			if (text === 'i' || text === 'a') {
				// bug in DeepSpeech 0.6 causes silence to be inferred as "i" or "a"
				return;
			}
			let recogTime = new Date().getTime() - start.getTime();
			return {
				text,
				recogTime,
				audioLength: Math.round(recordedAudioLength)
			};
		}
	}
	silenceBuffers = [];
	modelStream = null;
}

function intermediateDecode() {
	let results = finishStream();
	createStream();
	return results;
}

function feedAudioContent(chunk) {
	outfile.write(chunk);
	recordedAudioLength += (chunk.length / 2) * (1 / 16000) * 1000;
	// abuses that slice is only a view, so the length of the slice'd buffer is number of samples
	englishModel.feedAudioContent(modelStream, chunk.slice(0, chunk.length / 2));
}

let microphone;
function startMicrophone(callback) {
	if (microphone) {
		console.log('microphone exists');
		return;
	}
	
	createStream();
	
	microphone = mic({
		rate: '16000',
		channels: '1',
		debug: false,
		fileType: 'wav'
	});
	
	var stream = microphone.getAudioStream();
	
	stream.on('data', function(data) {
		processAudioStream(data, (results) => {
			callback(results);
		});
	});
	
	microphone.start();
}

function stopMicrophone() {
	microphone.stop();
	resetAudioStream();
}

function onRecognize(results) {
	if (results.text === 'quit') {
		console.log('quitting...');
		stopMicrophone();
		process.exit();
	}
	else {
		console.log('recognized:', results);
	}
}

if (process.argv[2]) {
	// if an audio file is supplied as an argument, play through the speakers to be picked up by the microphone
	console.log('play audio file', process.argv[2]);
	var file = fs.createReadStream(process.argv[2]);
	var reader = new wav.Reader();
	reader.on('format', function (format) {
		//firstChunkVoice = true;   // override vad for this test
		//SILENCE_THRESHOLD = 1000; // override silence (debounce time)
		// startMicrophone(function(results) {
		// 	console.log(results);
		// 	process.exit();
		// });
		// setTimeout(function() {
		// 	reader.pipe(new Speaker(format));
		// },900);
		console.log(format);
		len = format.length;
		console.log(len);
		for (i = 0; i < len; i += 32768)
		{
			data = format.slice(i, i + 32768);
			processAudioStream(data, (results) => {
				console.log(results);
			});
		}
	});
	file.pipe(reader);
}
else {
	startMicrophone(onRecognize);
}
