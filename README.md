# joi
Create your own talking chatbot. API infused from OpenAI and ElevenLabs and highly customizable.

### Preamble
So I've used all the major online LLMs out there. I've tried setting up my local models. But I thought to myself, why hasn't anybody made one of these talk to me like Gemini can do? So I decided to do it myself.

### Warning
This code simply makes calls to the OpenAI and ElevenLabs APIs for AI and speech generation, NEITHER OF WHICH ARE FREE. You will need BOTH a subscription to OpenAI and ElevenLabs API to use this code.

### Installation

Instructions only available for Windows currently because that's my native platform.

- Install Git for Windows from https://git-scm.com/downloads/win and use all the default options. Then restart your computer.
- Install Python for Windows from https://www.python.org/downloads/ and use all the default options. Then restart your computer.
- Open a terminal to a convenient location, I recommend your downloads directory. On Windows 11, you can go there and right click inside, then select "Open with Terminal."
- Type the following commands in order.

1. `git clone https://github.com/fordinator/joi`
2. `pip install os time whisper threading numpy sounddevice elevenlabs openai rich queue`
3. `cd joi`
4. `python joi_clean.py`

### Adding your API Keys

This code WILL NOT WORK without a PAID SUBSCRIPTION to BOTH ChatGPT's API and ElevenLab's API.

You can get a ChatGPT API subscription from https://platform.api.com. (I trust you know how to subscribe to things.) The way it works is you add a credit card and a small pre-payment ($10 is probably plenty) and OpenAI will deduct money from your balance every few messages you sent. Currently, the joi code is programmed to require about 3 cents per request.

You can get an Elevenlabs subscription from https://elevenlabs.io/pricing. The free account is okay for trying out joi but you will quickly run out of credits and will be forced to subscribe. Currently this code is very expensive on credits although I am researching ways to reduce that cost. The estimate is about 5 cents per request currently.

Once you have your subscriptions, you can create API keys for OpenAI at https://platform.openai.com/api-keys. Generate a code, give it a name (doesn't matter what it is) and once you have it SAVE IT SOMEWHERE so you don't forget it.

You can create API keys for ElevenLabs at https://elevenlabs.io/app/settings/api-keys. Generate a code, use the default name, and once you have it SAVE IT SOMEWHERE so you don't forget it.

Next you need to open `joi_clean.py` with a text editor (Notepad++ is good because its easier to read' and add these keys to the correct locations. You will NEED TO READ the instructions inside the code.

That's it! You can run `joi_clean.py` from a terminal now and listen to my default personality, caco-bot, insult you until you want to punch him.

### Customization

Further instructions for customization are in the comment sections of `joi_clean.py`. You can give the bot a different System Prompt so it has a personality entirely of your own design. You can select from among the 3,000+ voices available in the ElevenLabs Voice Library to give it exactly the voice you prefer.

Consult the comments of `joi_clean.py` for further details. YOU WILL NEED TO READ AND USE YOUR BRAIN.

Please none that due to the publicly available nature of OpenAI and ElevenLabs, usage of their services for illegal or sexual, violent or bigoted content is a violation of their terms of service and may result in legal action. I am considering modifying this code using Ollama and langchain to get around that but I lack the experience presently.

Watch this space!




