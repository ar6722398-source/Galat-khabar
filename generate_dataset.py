"""Builds a local, labelled demo dataset (data/news_dataset.csv).

The headlines are synthetic: written in the *style* of reliable reporting vs. misinformation.
For the final submission you can replace data/news_dataset.csv with a real dataset
(e.g. Kaggle 'Fake and Real News' / ISOT) that has columns: text,label,topic  (label = Real/Fake)
and run:  python train_model.py
"""
import random
import pandas as pd

random.seed(7)

TOPICS = {
    "Health": dict(
        subj=["a new vaccine", "a diabetes drug", "a dengue prevention programme", "a heart-health screening camp", "an antibiotic guideline"],
        real=["The health ministry said on {d} that {s} was approved after clinical trials involving {n} participants, according to an official statement.",
              "Researchers published a peer-reviewed study showing {s} reduced hospital admissions by {p} percent, the journal reported.",
              "According to data from the national health survey, {s} has reached {n} patients, a spokesperson said.",
              "Doctors at a government hospital reported that {s} showed mild side effects in {p} percent of cases, officials said."],
        fake=["SHOCKING!!! Doctors don't want you to know that {s} is a secret cover-up and a 100% miracle cure. Share before it is deleted!!!",
              "BREAKING: {s} secretly kills thousands and the government is hiding the hidden truth. Forward this to everyone you love!!!",
              "You won't believe what happened after drinking this home remedy instead of {s}. Big pharma is BANNING this video!",
              "WAKE UP people! {s} is a lie, a friend of my cousin who works inside confirmed it. Viral message, must watch!!!"]),
    "Politics": dict(
        subj=["the new education bill", "the state budget", "the election commission", "the transport policy", "the municipal reforms"],
        real=["Parliament passed {s} on {d} after a debate, the speaker's office said in a statement.",
              "According to the official gazette, {s} will come into effect next month, and the ministry released the full text.",
              "The opposition and ruling party members questioned {s} during a committee hearing, reported local newspapers citing official records.",
              "A government spokesperson said {s} was reviewed by an expert panel, with data on its cost published on the ministry website."],
        fake=["EXPOSED: {s} is a massive scam and leaders are laughing at you!!! Share this urgent news before they delete it!!!",
              "SECRET leaked document PROVES {s} was planned by foreign enemies. The media will never tell you this, spread the word!!!",
              "Shocking truth about {s}: everyone is lying to you. 100% proven by an unnamed insider, forward to 10 groups now!",
              "They don't want you to know what {s} really means. WAKE UP before it is too late!!!"]),
    "Technology": dict(
        subj=["a popular messaging app", "the new smartphone", "a government digital payment system", "an AI chatbot", "the 5G rollout"],
        real=["The company announced on {d} that {s} received a security update after researchers reported a bug, according to its blog.",
              "A study by university engineers found {s} processed {n} requests per second in independent tests, the report said.",
              "Officials said {s} now serves {n} users, based on figures released in the quarterly report.",
              "The regulator published guidelines for {s}, and a company spokesperson confirmed compliance in a statement."],
        fake=["URGENT!!! Forward this message or {s} will start charging you money tomorrow. Confirmed by a secret source!!!",
              "WARNING: {s} is secretly recording everything and sending it to hackers. Share this with all your contacts immediately!!!",
              "You won't believe it: {s} gives FREE data to everyone who clicks this link. Viral offer, 100% real, limited time!!!",
              "Hidden truth about {s}: scientists are BANNED from telling you it causes cancer. Share before deleted!!!"]),
    "Science": dict(
        subj=["a deep-space telescope", "the monsoon forecast", "an earthquake early-warning system", "a new species of frog", "a solar mission"],
        real=["Scientists reported that {s} recorded new data on {d}, and the findings were published in a peer-reviewed journal.",
              "The space agency said {s} completed its test successfully, according to data released by the mission team.",
              "According to the meteorological department, {s} predicts {p} percent of normal rainfall, officials said in a briefing.",
              "A research team at the institute said {s} was verified by independent experts, the study noted."],
        fake=["MIRACLE discovery: {s} PROVES that NASA lied to you for years!!! Secret scientists exposed the cover-up, share now!!!",
              "BREAKING!!! {s} will cause the end of the world next week, a viral message from an anonymous expert warns.",
              "Scientists HATE this: {s} is a hoax and the hidden truth is being banned online. Must watch before deleted!!!",
              "Shocking: {s} secretly controls the weather and nobody will tell you. Forward this to 20 people!"]),
    "Sports": dict(
        subj=["the national cricket team", "the football league final", "the athletics championship", "a star batter", "the hockey squad"],
        real=["The board said on {d} that {s} will play {n} matches this season, according to the official schedule.",
              "Officials reported that {s} won by {p} runs, and the match report was published by the organisers.",
              "A spokesperson for the federation said {s} completed fitness tests, as stated in the press release.",
              "According to the tournament data, {s} recorded its best score in {n} days, the committee said."],
        fake=["SHOCKING: {s} match was FIXED and the hidden truth is being deleted!!! Share before they ban this post!!!",
              "You won't believe the secret about {s}: insider says everything is a scam, forward this to your friends!!!",
              "BREAKING!!! {s} banned for life, 100% confirmed by a viral message. Wake up, fans, they are lying to you!!!",
              "EXPOSED: {s} secretly took a miracle drug. Doctors hate this, share it NOW!!!"]),
}

DATES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "12 March", "5 June", "21 August"]
NOISE_REAL = ["Breaking: ", "Update: ", "Urgent notice: ", ""]
NOISE_FAKE = ["", "", "Reportedly, ", "According to some people, "]


def make(template, subj):
    return template.format(s=subj, d=random.choice(DATES), n=random.choice([120, 850, 2400, 15000, 60000]),
                           p=random.choice([4, 9, 12, 18, 27, 35]))

rows = []
for topic, t in TOPICS.items():
    for _ in range(60):
        rows.append((random.choice(NOISE_REAL) + make(random.choice(t["real"]), random.choice(t["subj"])), "Real", topic))
        txt = random.choice(NOISE_FAKE) + make(random.choice(t["fake"]), random.choice(t["subj"]))
        if random.random() < 0.15:   # a few subtle fakes written in a calm tone -> keeps the task realistic
            txt = f"A message circulating online claims that {random.choice(t['subj'])} was secretly cancelled, but nobody has confirmed it."
        rows.append((txt, "Fake", topic))

df = pd.DataFrame(rows, columns=["text", "label", "topic"]).sample(frac=1, random_state=1).reset_index(drop=True)
df.to_csv("data/news_dataset.csv", index=False)
print(df.label.value_counts().to_string(), "\nrows:", len(df))
