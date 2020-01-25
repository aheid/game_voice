# NodeJS Game Voice test client

This is a NodeJS example of recording from the microphone and streaming to
DeepSpeech with voice activity detection, and sending commands via UART to HID input generator.

Based off nodejs_mic_vad_streaming from https://github.com/mozilla/DeepSpeech-examples/

Modified mic library to reduce stream chunk size and workaround for sox pipe bug.

### Prerequisites:

1) The example utilized the [mic](https://github.com/ashishbajaj99/mic) NPM module which requires
either [sox](http://sox.sourceforge.net/) (Windows/Mac) or [arecord](http://alsa-project.org/) (Linux).

2) Download the pre-trained DeepSpeech english model (1.8GB):

```
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.6.0/deepspeech-0.6.0-models.tar.gz
tar xvfz deepspeech-0.6.0-models.tar.gz
```

#### Install:

```
npm install
```

#### Run NodeJS server:

```
node start.js
```

#### Specify alternate DeepSpeech model path:

Use the `DEEPSPEECH_MODEL` environment variable to change models.
Use the `DEEPSPEECH_LM` environment variable to change langauge model.

```
DEEPSPEECH_MODEL=~/dev/jaxcore/deepspeech-0.6.0-models/ node start.js
```