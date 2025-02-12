# joi
Create your own talking chatbot. API infused from OpenAI and ElevenLabs and highly customizable.

(Inspired and heavily derived from both https://github.com/vndee/local-talking-llm and https://github.com/jakobdylanc/llmcord)

### Preamble
So I've used all the major online LLMs out there. I've tried setting up my local models. But I thought to myself, why hasn't anybody made one of these talk to me like Gemini can do? So I decided to do it myself.

### Warning
This code simply makes calls to the OpenAI and ElevenLabs APIs for AI and speech generation, NEITHER OF WHICH ARE FREE. You will need BOTH a subscription to OpenAI and ElevenLabs API to use this code.

### Prerequisites

You will need a microphone to interact with joi. You talk to her and she talks to you. Most laptops have one included and preconfigured but desktop users will probably have to buy one. A clip mic or a gaming headset will work in a pinch.

Just MAKE SURE ITS PLUGGED IN and selected as your "DEFAULT INPUT DEVICE".

### Installation

Instructions only available for Windows currently because that's my native platform.

- Install Git for Windows from https://git-scm.com/downloads/win and use all the default options. Then restart your computer.
- Install Python for Windows from https://www.python.org/downloads/ and use all the default options. Then restart your computer.
- Open a terminal to a convenient location, I recommend your downloads directory. On Windows 11, you can go there and right click inside, then select "Open with Terminal."
- Type the following commands in order.

1. `git clone https://github.com/fordinator/joi`
2. `cd joi`
3. `pip install os time whisper threading numpy sounddevice elevenlabs openai rich queue`
4. `python joi_clean.py`

### Adding your API Keys

This code WILL NOT WORK without a PAID SUBSCRIPTION to BOTH ChatGPT's API and ElevenLab's API.

You can get a ChatGPT API subscription from https://platform.api.com. (I trust you know how to subscribe to things.) The way it works is you add a credit card and a small pre-payment ($10 is probably plenty) and OpenAI will deduct money from your balance every few messages you sent. Currently, the joi code is programmed to require about 3 cents per request.

You can get an ElevenLabs subscription from https://elevenlabs.io/pricing. The free account is okay for trying out joi but you will quickly run out of credits and will be forced to subscribe. Currently this code is very expensive on credits although I am researching ways to reduce that cost. The estimate is about 5 cents per request currently.

Once you have your subscriptions, you can create API keys for OpenAI at https://platform.openai.com/api-keys. Generate a code, give it a name (doesn't matter what it is) and once you have it SAVE IT SOMEWHERE so you don't forget it.

You can create API keys for ElevenLabs at https://elevenlabs.io/app/settings/api-keys. Generate a code, use the default name, and once you have it SAVE IT SOMEWHERE so you don't forget it.

Next you need to open `joi_clean.py` with a text editor (Notepad++ is good because its easier to read) and add these keys to the correct locations. You will NEED TO READ the instructions inside the code.

### Adding Eastend Steve

One final step. You need to add a voice to your ElevenLabs Library or you WILL RECEIVE AN ERROR when you try to run this code. Here are extremely detailed instructions on how to do so.

- From https://elevenlabs.io/, sign in and click "Go To App" in the upper right.
- Click `Voices+` in the left menu.
- Click `Community` along the middle.
- Search for "Eastend Steve"
- Click `View` on "Eastend Steve"
- Click "Add To My Voices"

Optionally you can pick your own voice. Consult the comments of `joi_clean.py` for details.

That's it! You can run `joi_clean.py` from a terminal now and listen to my default personality, caco-bot, insult you until you want to punch him.

### Variants

If you would prefer to use the DIRT CHEAP MistralAI API instead of the OpenAI API, download and run the `joi_clean_mistral.py` instead of `joi_clean.py`, following all the same steps and instructions.

Oh, you will need to do `pip install mistralai` inside the joi directory in addition to the other steps.

And of course, READ THE COMMENTS of `joi_clean_mistral.py` to input the proper MistralAI API key.

For an ABSOLUTE BARGAIN BASEMENT option, you can use MistralAI with PlayHT. This one is not very smart and the voices are more robotic, but it's an option if your budget is really constrained and you want to talk to the bot for a long time. Download the `joi_clean_mistral_playht.py` file and run it same as usual.

You will need to do the `pip install mistralai` as above, along with `pip install io select wave.`

And don't forget to read the comments!

### Customization

Further instructions for customization are in the comment sections of `joi_clean.py`. You can give the bot a different System Prompt so it has a personality entirely of your own design. You can select from among the 3,000+ voices available in the ElevenLabs Voice Library to give it exactly the voice you prefer.

Consult the comments of `joi_clean.py` for further details. YOU WILL NEED TO READ AND USE YOUR BRAIN.

Please none that due to the publicly available nature of OpenAI and ElevenLabs, usage of their services for illegal or sexual, violent or bigoted content is a violation of their terms of service and may result in legal action. 

However I keep a `joi_dirty.py` script on my Discord that you can use to bypass some of these restrictions. You'll just have to join and ask me for it. It is... a bit more complicated to set up.

### Support

Support is not guaranteed but submitting a Github issue is preferred. I keep a currently empty Discord at https://discord.gg/gAugxKBHQY if you want to stop by and shoot the shit. I am known as "\_.vexation.\_"

### Uninstallation

Just delete the joi folder and uninstall Git for Windows and Python for Windows, even though you probably shouldn't. They're very useful tools.

### Contributing

This code is just a wrapper for OpenAI and ElevenLabs. Steal it, modify it, sell it, I don't give a fuck. If you could drop me a dollar at https://patreon.com/cacophony1979 it would go a long way to encouraging me to continue with projects like this. 

Future plans include implementing Llama, Groq, Mistral and other LLMs along with an endless number of local models for roleplaying scenarios but that takes time, and time is money.

### Final CODA

Stay real you horny bastards.




