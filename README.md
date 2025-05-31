# joi - Configurable Voice Chatbot for the Masses

```
      ,--.-,   _,.---._     .=-.-. 
     |==' -| ,-.' , -  `.  /==/_ / 
     |==|- |/==/_,  ,  - \|==|, |  
   __|==|, |==|   .=.     |==|  |  
,--.-'\=|- |==|_ : ;=:  - |==|- |  
|==|- |=/ ,|==| , '='     |==| ,|                      
|==|. /=| -|\==\ -    ,_ /|==|- |          
\==\, `-' /  '.='. -   .' /==/. /  
 `--`----'     `--`--''   `--`-`  
```

Create your own talking chatbot. Using APIs from multiple LLMs and text-to-speech services.

(Inspired and heavily derived from both https://github.com/vndee/local-talking-llm and https://github.com/jakobdylanc/llmcord)

### Preamble
So I've used all the major online LLMs out there. I've tried setting up my local models. But I thought to myself, why hasn't anybody made one of these talk to me like Gemini can do? So I decided to do it myself.

### Warning
This code simply makes calls to populal LLM and TTS APIs for AI and speech generation, NEITHER OF WHICH ARE FREE. You will need BOTH a subscription to an LLM API **AND** a TTS API to use this.

Recently an option has been provided to set up the local models Ollama for LLM and Kokoro for TTS but that is outside the scope of this document.

### Prerequisites

The BEST way to interact with joi is by microphone. You talk to her and she talks to you. Most laptops have one included and preconfigured but desktop users will probably have to buy one. A clip mic or a gaming headset will work in a pinch.

Just MAKE SURE ITS PLUGGED IN and selected as your "DEFAULT INPUT DEVICE".

As a fallback you can just type to her, but I find that less compelling.

### Installation

Instructions only available for Windows currently because that's my native platform.

- Install Git for Windows from https://git-scm.com/downloads/win and use all the default options. Then restart your computer.
- Install Python for Windows from https://www.python.org/downloads/ and use all the default options. Then restart your computer.
- Open a terminal to a convenient location, I recommend your downloads directory. On Windows 11, you can go there and right click inside, then select "Open with Terminal."
- Type the following command: `pip install -r requirements.txt'

### Adding your API Keys

This code WILL NOT WORK without a PAID SUBSCRIPTION to BOTH an LLM API and a TTS API. For LLM you can choose OpenAI, Gemini, Anthropic, or Mistral. For TTS you can choose ElevenLabs, PlayHT, Google Voice, or Azure. If you are familiar with setting up local models (which are free and do not send any information to tech bros so they can fire up _SkyNet_) there are now options for Ollama for the LLM and Kokoro for the TTS.

The default instructions assume the most advanced services: OpenAI and ElevenLabs. Additional instructions will not be documented, but you can always find me and ask me.

You can get a ChatGPT API subscription from https://platform.api.com. (I trust you know how to subscribe to things.) The way it works is you add a credit card and a small pre-payment ($10 is probably plenty) and OpenAI will deduct money from your balance every few messages you sent. Currently, the joi code is programmed to require about 3 cents per request. 

You can get an ElevenLabs subscription from https://elevenlabs.io/pricing. The free account is okay for trying out joi but you will quickly run out of credits and will be forced to subscribe. Currently this code is very expensive on credits so you may wish to investigate other TTS APIs or just use Kokoro. The estimate is about 5 cents per request currently. 

Once you have your subscriptions, you can create API keys for OpenAI at https://platform.openai.com/api-keys. Generate a code, give it a name (doesn't matter what it is) and once you have it SAVE IT SOMEWHERE so you don't forget it.

You can create API keys for ElevenLabs at https://elevenlabs.io/app/settings/api-keys. Generate a code, use the default name, and once you have it SAVE IT SOMEWHERE so you don't forget it.

Next you need to open `joi.py` with a text editor (Notepad++ is good because its easier to read) and add these keys to the correct locations. You will NEED TO READ the instructions inside the code.

### Adding Vexation

One final step. You need to add a voice to your ElevenLabs Library or you WILL RECEIVE AN ERROR when you try to run this code. Here are extremely detailed instructions on how to do so.

- From https://elevenlabs.io/, sign in and click "Go To App" in the upper right.
- Click `Voices+` in the left menu.
- Click `Community` along the middle.
- Search for "Vexation"
- Click `View` on "Vexation"
- Click "Add To My Voices"

Optionally you can pick your own voice. Consult the comments of `joi.py` for details.

That's it! You can run `joi.py` from a terminal now and listen to my default personality, caco-bot, insult you until you want to punch him.

### Customization

Further instructions for customization are in the comment sections of `joi.py`. You can give the bot a different System Prompt so it has a personality entirely of your own design. You can select from among the 3,000+ voices available in the ElevenLabs Voice Library to give it exactly the voice you prefer.

Consult the comments of `joi.py` for further details. YOU WILL NEED TO READ AND USE YOUR BRAIN.

Please note that due to the publicly available nature of OpenAI and ElevenLabs, usage of their services for illegal or sexual, violent or bigoted content is a violation of their terms of service and may result in legal action. 

For that you will have to use a local model which is available in the code. But setting one up is your problem.

### Support

Support is not guaranteed but submitting a Github issue is preferred.

### Uninstallation

Just delete the joi folder and uninstall Git for Windows and Python for Windows, even though you probably shouldn't. They're very useful tools.

### Contributing

This code is just a wrapper for OpenAI and ElevenLabs. Steal it, modify it, sell it, I don't care. If you could drop me a dollar at https://patreon.com/vexation1977 it would go a long way to encouraging me to continue with projects like this. 

### Licensing and Usage

Steal it, take credit for it, change it, improve it, sell it, it's yours.

If you want to bullshit with me, visit https://discord.com/invite/gAugxKBHQY.

If you want to encourage other coding projects like this, donate to https://www.patreon.com/vexation1977
