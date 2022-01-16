import matplotlib.pyplot as plt
from collections import defaultdict
import os
import itertools
from baseline import feature_mapping
import pandas as pd
import itertools
from nltk.corpus import stopwords
import nltk

function_word_tags = {"CC", "CD", "DT", "EX", "IN", "LS", "POS", "PDT", "PRP", "PRP$", "RP", "TO", "UH", "WDT", "WP",
                      "WP$", "WRB"}

sentiment_features = ['negative_adjectives_component', 'social_order_component', 'action_component',
                      'positive_adjectives_component',
                      'joy_component', 'affect_friends_and_family_component', 'fear_and_digust_component',
                      'politeness_component',
                      'polarity_nouns_component', 'polarity_verbs_component', 'virtue_adverbs_component',
                      'positive_nouns_component',
                      'respect_component', 'trust_verbs_component', 'failure_component', 'well_being_component',
                      'economy_component',
                      'certainty_component', 'positive_verbs_component', 'objects_component']


# read the attribution file
def read_word_attributions(path, sign_score, positive, word2attributions, sign_word_dic):
    for line in open(path, "r"):
        parts = line.strip().split("\t")
        predicted_label = parts[3]
        if predicted_label not in word2attributions:
            word2attributions[predicted_label] = defaultdict(list)
        if predicted_label not in sign_word_dic:
            sign_word_dic[predicted_label] = defaultdict(int)
        attributions = parts[4]
        attributions = [el for el in attributions.replace("[", "").replace("]", "").split(", (")]
        attributions = [el.split(",") for el in attributions]
        for att in attributions:
            word = att[0]
            word = word.replace("'", "").strip()
            score = att[1]
            score = score.replace(")", "").strip()
            if score == "'":
                continue
            score = float(score)
            if positive:
                if score >= sign_score:
                    word2attributions[predicted_label][word].append(score)
                    sign_word_dic[predicted_label][word] += 1
            else:
                if score <= sign_score:
                    word2attributions[predicted_label][word].append(score)
                    sign_word_dic[predicted_label][word] += 1

    return word2attributions, sign_word_dic


def get_all_attribution_scores(files):
    all_attributions = []
    for file in files:
        for line in open(file, "r"):
            parts = line.strip().split("\t")
            attributions = parts[4]
            attributions = [el for el in attributions.replace("[", "").replace("]", "").split(", (")]
            attributions = [el.split(",")[1].strip().replace(")", "") for el in attributions]
            attributions = [el for el in attributions if el != "'"]
            for el in attributions:
                all_attributions.append(float(el))
    return all_attributions


def plot_attributions(attributions):
    plt.hist(attributions, bins=30)
    plt.show()


def get_files(dimension):
    attributions = "/Users/falkne/PycharmProjects/DQI/results/attribution_results"
    files = os.listdir(attributions)
    dimension_files = []
    for f in files:
        if dimension in f:
            dimension_files.append("%s/%s" % (attributions, f))
    return dimension_files


def get_dictionaries(files):
    word2attributions = {}
    word2counts = {}
    for f in files:
        word2attributions, word2counts = read_word_attributions(f, 0.1, positive=True,
                                                                word2attributions=word2attributions,
                                                                sign_word_dic=word2counts)
    sorted_dic = {}
    for k, v in word2counts.items():
        sorted_dic[k] = dict(sorted(v.items(), key=lambda item: item[1], reverse=True))
    return word2attributions, sorted_dic


def write_files_for_seance(dictionary, dimension):
    for label, words in dictionary.items():
        os.makedirs("/Users/falkne/PycharmProjects/DQI/data/test_seance/%s/%s" % (dimension, label))
        for word in words.keys():
            with open("/Users/falkne/PycharmProjects/DQI/data/test_seance/%s/%s/%s.txt" % (dimension, label, word),
                      "w") as out:
                out.write(word)


def extract_feature_dist(dimension, labels):
    path = "/Users/falkne/PycharmProjects/DQI/data/test_seance/%s" % dimension
    for label in labels:
        data = pd.read_csv("%s/%s/results.csv" % (path, label))
        data = data[sentiment_features]
        data.rename(feature_mapping, inplace=True, axis=1)
        data.loc['mean'] = data.mean()
        label2feature_means = dict(zip(data.columns, data.loc["mean"].values))
        label2feature_means = dict(sorted(label2feature_means.items(), key=lambda item: item[1], reverse=True))
        print(label)
        print("####")
        for feat, score in label2feature_means.items():
            print(feat, score)


def get_distinct_words(dictionary):
    new_dic = {}
    for label, words in dictionary.items():
        all_other_words = [list(dictionary[l].keys()) for l in dictionary.keys() if l != label]
        all_other_words = set(list(itertools.chain(*all_other_words)))
        for word, freq in words.items():
            if word not in all_other_words:
                if label not in new_dic:
                    new_dic[label] = {}
                new_dic[label][word] = freq
    return new_dic


def get_dics_by_pos_tag(dictionary):
    important_function_words = {}
    important_content_words = {}
    for label, words in dictionary.items():
        if label not in important_function_words:
            important_function_words[label] = {}
        if label not in important_content_words:
            important_content_words[label] = {}
        for word, freq in words.items():
            pos_tag = nltk.pos_tag([word])[0][1]
            if pos_tag in function_word_tags:
                important_function_words[label][word] = freq
            else:
                important_content_words[label][word] = freq

    return important_function_words, important_content_words


def extract_good_examples(files, important_words, target_label):
    max_score = 0
    max_sent = ""
    for file in files:
        for line in open(file):
            parts = line.strip().split("\t")
            predicted_label = parts[3]
            if predicted_label == target_label:
                attributions = parts[4]
                attributions = [el for el in attributions.replace("[", "").replace("]", "").split(", (")]
                attributions_scores = [el.split(",")[1].strip().replace(")", "") for el in attributions]
                attributions_scores = [float(el) for el in attributions_scores if el != "'"]
                words = [el.split(",")[0].replace("'", "").strip().replace("(", "") for el in attributions]
                if len(set(words).intersection(set(important_words))) > 5:
                    print(line)
                    sum_scores = sum(attributions_scores)
                    if sum_scores > max_score:
                        max_score = sum_scores
                        max_sent = line
    print(max_sent)


def get_split_target_sent(sent, dimension):
    for i in range(0, 5):
        test = pd.read_csv(
            "/Users/falkne/PycharmProjects/DQI/data/5foldAugmentedEDA/%s/split%d/test.csv" % (dimension, i), sep="\t")
        if sent in test.cleaned_comment.values:
            print("split %i" % i)


if __name__ == '__main__':
    interactivity = get_files("int")
    commongood = get_files("jcon")
    justification = get_files("jlev")
    respect = get_files("respect")

    files = list(itertools.chain.from_iterable([interactivity, commongood, justification, respect]))
    all_att_scores = get_all_attribution_scores(files)
    all_att_scores = list(filter(lambda x: x <= -0.10 or x >= 0.10, all_att_scores))

    jlev_attributions, jlev_counts = get_dictionaries(justification)
    jcon_attributions, jcon_counts = get_dictionaries(commongood)
    int_attributions, int_counts = get_dictionaries(interactivity)
    respect_attributions, respect_counts = get_dictionaries(respect)
    print(jcon_counts.keys())
    # extract_good_examples(commongood, list(jcon_counts["common good"].keys()), 'common good')
    # target_sent = "Strangely enough, I am in favor of strict border control, because illegal migration compromises the position of migrants who are already in Europe. So you have to see that this illegal migration, the reason for illegal migration, has to be eliminated. These are two reasons why people come. Because there is an opportunity for work and because there are too few people in Europe to do certain jobs. And so if you are in the EU and you want to stay there, you have to see that employers don't have that kind of need so that the market for illegal immigrants is finally dried up."
    # target_sent = "I think that we should know - we should not let people arrive en masse. I think it's also something about immigrants who arrive, who have a work permit afterwards that is not being deported. If these people, for example, give satisfaction, pay taxes, go to school, I don't see why we don't renew their work permit. Because we accepted them, they stayed but so much I don't know, or less, well, all is going very well, why they - as it happened in France, the cause was absolutely indisputable [?], the child was schooled and they sent them back, the parents and the grandfather, to his country of origin, I find that absolutely monstrous. Because they are people who did not hinder, they even brought to the society, I am completely for the immigration because there is a richness to see in all the immigrants. I am not saying that everything is rosy, that all is easy and there are immigrants who are more difficult than others to control, but I believe that it is a wealth in the country and our country - France - is an old country and it is time that there is some new blood and people who make children and not only centenarians. I'm all for immigration."
    # target_sent = "Yes, two examples of values: I think in Western Europe, for example, I take the image of women or I take the point of view of revenge, which is clearly differently formed and translated into norms in the realm of Islam and also in their peoples than in ours. I do believe that there Europe and Islam are clearly different. Thank you"
    # target_sent = "I would just like to say that if you look at the whole thing in Germany, that's all well and good with the language, but these parallel societies with the Muslims, that bothers me enormously in Germany. And if it goes on like this, I'm afraid that my grandchildren will get a headscarf. Yes! I see it that way! because they are so fanatical. All over the world! Just look at them! These fundamentalists. What are they doing for... so suicide bombers and everything... it really upsets me! That's right! Look at the German cities! Yes. Real parallel societies. There are whole districts that are dominated only by Turks."
    # target_sent = "I agree exactly with what JP just said, in addition to the fact that in Belgium, the Chinese are settling more and more, in catering, in construction and always illegally, and always to be paid less - they - Yes, that the boss pays less. I agree with JP exactly."
    # target_sent = "Well, I agree with all those who had a say, but what is important for me is that also the people who are here illegally and who have very often, for example, health problems and no money to go to see a doctor - This is really nothing good - neither for them nor for their families and it is a reason for suffering."
    # target_sent = "Thank you. I don't want to push myself forward, but I also think that the EU is very important and challenged and the richer countries of the EU simply have to help those who have now joined us very quickly, such as Bulgaria and Romania, for example, yes? We have an increase in crime in Vienna, yes? It is so high, in the last month, I think, 30 percent more cars were stolen. Only in the last month, yes? And I understand the people. I mean, I wouldn't be happy if my car was gone either, but they come and see that there's so much of it here, right? And with them there is nothing left. And after communism, everything collapsed in these two countries, for example. And anyone who has been there will also see that agriculture no longer exists there. That used to be the granary of the Austrian monarchy, yes? And nothing exists anymore. The fields are lying fallow, they are becoming desolate. And at most, private households do a little bit for themselves in a tiny little garden, yes? But this is where the EU is really needed with programs, because we can use the things that are produced down there 100 percent. I don't need anything from South America, or I don't need fruit from South Africa, when I can have Bulgaria and Romania, yes? And I think Germany has been so efficient with the colonization, let's just say it, of the new federal states, yes? They have built up so much in no time at all. And they were also doing very badly, yes, when they opened up. And now 20 years later, it's exactly 20 years ago, it's a flourishing country, yes? And I think the entire EU should do the same with the other countries in the East. Takes time, also takes time, but [grins]. So, that's it again, sorry."
    # target_sent = "As far as the discussion is concerned; we have been saying that within the EU, there are countries where the economic development is far lower than in other countries. That is why the EU funds could be invested in those less developed countries, in order to make their chances equal to everyone. That would mean that EU countries should contribute a part of the generated wealth and welfare to the poorer."
    # get_split_target_sent(target_sent, "jcon")
    # jlev_counts = get_distinct_words(jlev_counts)
    file = open("results/respect_function_words.csv", "w")
    jlev_function_words, jlev_content_words = get_dics_by_pos_tag(respect_counts)
    for label, words in jlev_function_words.items():
        print(label, list(words)[:10])
        file.write(label + "\t" + ", ".join(list(words)[:10]) + "\n")
    file.close()
    print("\n")
    file = open("results/respect_content_words.csv", "w")
    for label, words in jlev_content_words.items():
        print(label, list(words)[:20])
        file.write(label + "\t" + ", ".join(list(words)[:20]) + "\n")
    file.close()
    # file = open("results/jlev_important_words.csv", "w")
    # for label, words in jlev_counts.items():
    #    top_25 = list(words.keys())[:25]
    #    file.write(label + "\t" + ", ".join(top_25) + "\n")
    # file.close()
