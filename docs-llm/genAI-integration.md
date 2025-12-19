Google GenAI
In this notebook, we show how to use the google-genai Python SDK with LlamaIndex to interact with Google GenAI models.

If you’re opening this Notebook on colab, you will need to install LlamaIndex 🦙 and the google-genai Python SDK.

%pip install llama-index-llms-google-genai llama-index

Basic Usage
You will need to get an API key from Google AI Studio. Once you have one, you can either pass it explicity to the model, or use the GOOGLE_API_KEY environment variable.

import os

os.environ["GOOGLE_API_KEY"] = "..."

Basic Usage
You can call complete with a prompt:

from llama_index.llms.google_genai import GoogleGenAI

llm = GoogleGenAI(
    model="gemini-2.5-flash",
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)

resp = llm.complete("Who is Paul Graham?")
print(resp)

Paul Graham is a prominent figure in the tech world, best known for his work as a programmer, essayist, and venture capitalist. Here's a breakdown of his key contributions:

*   **Programmer and Hacker:** He's a skilled programmer, particularly in Lisp. He co-founded Viaweb, which was one of the first software-as-a-service (SaaS) companies, providing tools for building online stores. Yahoo acquired Viaweb in 1998, and it became Yahoo! Store.

*   **Essayist:** Graham is a prolific and influential essayist. His essays cover a wide range of topics, including startups, programming, design, and societal trends. His writing style is known for being clear, concise, and thought-provoking. Many of his essays are considered essential reading for entrepreneurs and those interested in technology.

*   **Venture Capitalist and Founder of Y Combinator:** Perhaps his most significant contribution is co-founding Y Combinator (YC) in 2005. YC is a highly successful startup accelerator that provides seed funding, mentorship, and networking opportunities to early-stage startups. YC has funded many well-known companies, including Airbnb, Dropbox, Reddit, Stripe, and many others. Graham stepped down from his day-to-day role at YC in 2014 but remains involved.

In summary, Paul Graham is a multifaceted individual who has made significant contributions to the tech industry as a programmer, essayist, and venture capitalist. He is particularly known for his role in founding and shaping Y Combinator, one of the world's leading startup accelerators.

You can also call chat with a list of chat messages:

from llama_index.core.llms import ChatMessage
from llama_index.llms.google_genai import GoogleGenAI

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]
llm = GoogleGenAI(model="gemini-2.5-flash")
resp = llm.chat(messages)

print(resp)

assistant: Ahoy there, matey! Gather 'round, ye landlubbers, and listen to a tale that'll shiver yer timbers and curl yer toes! This be the story of One-Eyed Jack's Lost Parrot and the Great Mango Mayhem!

Now, One-Eyed Jack, bless his barnacle-encrusted heart, was a fearsome pirate, alright. He could bellow louder than a hurricane, swing a cutlass like a dervish, and drink rum like a fish. But he had a soft spot, see? A soft spot for his parrot, Polly. Polly wasn't just any parrot, mind ye. She could mimic the captain's every cuss word, predict the weather by the way she ruffled her feathers, and had a particular fondness for shiny trinkets.

One day, we were anchored off the coast of Mango Island, a lush paradise overflowing with the juiciest, sweetest mangoes ye ever did see. Jack, bless his greedy soul, decided we needed a cargo hold full of 'em. "For scurvy prevention!" he declared, winking with his good eye. More like for his own personal mango-eating contest, if ye ask me.

We stormed ashore, cutlasses gleaming, ready to plunder the mango groves. But Polly, the little feathered devil, decided she'd had enough of the ship. She squawked, "Shiny! Shiny!" and took off like a green streak towards the heart of the island.

Jack went ballistic! "Polly! Polly, ye feathered fiend! Get back here!" He chased after her, bellowing like a lovesick walrus. The rest of us, well, we were left to pick mangoes and try not to laugh ourselves silly.

Now, Mango Island wasn't just full of mangoes. It was also home to a tribe of mischievous monkeys, the Mango Marauders, they were called. They were notorious for their love of pranks and their uncanny ability to steal anything that wasn't nailed down.

Turns out, Polly had landed right in the middle of their territory. And those monkeys, they took one look at her shiny feathers and decided she was the perfect addition to their collection of stolen treasures. They snatched her up, chattering and screeching, and whisked her away to their hidden lair, a giant mango tree hollowed out by time.

Jack, bless his stubborn heart, followed the sound of Polly's squawks. He hacked through vines, dodged falling mangoes, and even wrestled a particularly grumpy iguana, all in pursuit of his feathered friend.

Finally, he reached the mango tree. He peered inside and saw Polly, surrounded by a horde of monkeys, all admiring her shiny feathers. And Polly? She was having the time of her life, mimicking the monkeys' chattering and stealing their mangoes!

Jack, instead of getting angry, started to laugh. A hearty, booming laugh that shook the very foundations of the tree. The monkeys, startled, dropped their mangoes and stared at him.

Then, Polly, seeing her captain, squawked, "Rum! Rum for everyone!"

And that, me hearties, is how One-Eyed Jack ended up sharing a barrel of rum with a tribe of mango-loving monkeys. We spent the rest of the day feasting on mangoes, drinking rum, and listening to Polly mimic the monkeys' antics. We even managed to fill the cargo hold with mangoes, though I suspect a good portion of them were already half-eaten by the monkeys.

So, the moral of the story, me lads? Even the fiercest pirate has a soft spot, and sometimes, the best treasures are the ones you least expect. And always, ALWAYS, keep an eye on yer parrot! Now, who's for another round of grog?

Streaming Support
Every method supports streaming through the stream_ prefix.

from llama_index.llms.google_genai import GoogleGenAI

llm = GoogleGenAI(model="gemini-2.5-flash")

resp = llm.stream_complete("Who is Paul Graham?")
for r in resp:
    print(r.delta, end="")

Paul Graham is a prominent figure in the tech world, best known for his work as a computer programmer, essayist, venture capitalist, and co-founder of the startup accelerator Y Combinator. Here's a breakdown of his key accomplishments and contributions:

*   **Computer Programmer and Author:** Graham holds a Ph.D. in computer science from Harvard University. He is known for his work on Lisp, a programming language, and for developing Viaweb, one of the first software-as-a-service (SaaS) companies, which was later acquired by Yahoo! and became Yahoo! Store. He's also the author of several influential books on programming and entrepreneurship, including "On Lisp," "ANSI Common Lisp," "Hackers & Painters," and "A Plan for Spam."

*   **Essayist:** Graham is a prolific essayist, writing on a wide range of topics including technology, startups, art, philosophy, and society. His essays are known for their insightful observations, clear writing style, and often contrarian viewpoints. They are widely read and discussed in the tech community. You can find his essays on his website, paulgraham.com.

*   **Venture Capitalist and Y Combinator:** Graham co-founded Y Combinator (YC) in 2005 with Jessica Livingston, Robert Morris, and Trevor Blackwell. YC is a highly successful startup accelerator that provides seed funding, mentorship, and networking opportunities to early-stage startups. YC has funded many well-known companies, including Airbnb, Dropbox, Reddit, Stripe, and many others. While he stepped down from day-to-day operations at YC in 2014, his influence on the organization and the startup ecosystem remains significant.

In summary, Paul Graham is a multifaceted individual who has made significant contributions to computer science, entrepreneurship, and the broader tech culture. He is highly regarded for his technical expertise, insightful writing, and his role in shaping the modern startup landscape.

from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="user", content="Who is Paul Graham?"),
]

resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")

Paul Graham is a prominent figure in the tech world, best known for his work as a programmer, essayist, and venture capitalist. Here's a breakdown of his key contributions:

*   **Programmer and Hacker:** He is a skilled programmer, particularly in Lisp. He co-founded Viaweb, one of the first software-as-a-service (SaaS) companies, which was later acquired by Yahoo! and became Yahoo! Store.

*   **Essayist:** Graham is a prolific and influential essayist, writing on topics ranging from programming and startups to art, philosophy, and social commentary. His essays are known for their clarity, insight, and often contrarian viewpoints. They are widely read and discussed in the tech community.

*   **Venture Capitalist:** He co-founded Y Combinator (YC) in 2005, a highly successful startup accelerator. YC has funded and mentored numerous well-known companies, including Airbnb, Dropbox, Reddit, Stripe, and many others. Graham's approach to early-stage investing and startup mentorship has had a significant impact on the startup ecosystem.

In summary, Paul Graham is a multifaceted individual who has made significant contributions to the tech industry as a programmer, essayist, and venture capitalist. He is particularly influential in the startup world through his work with Y Combinator.

