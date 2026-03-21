"""
Prediction script for CSC311 proectt

Regan Quartus, Brenden McFarlane, Elise Corbin
March 26, 2025
"""
import sys
import csv
import statistics as st
import math
import numpy as np
import pandas as pd

# SIMILARITY VECTOR WEIGHTS
CLOCK_FEEL = {}
CLOCK_FOOD = {}
CLOCK_MUSIC = {}
LILIES_FEEL = {}
LILIES_FOOD = {}
LILIES_MUSIC = {}
STARRY_FEEL = {}
STARRY_FOOD = {}
STARRY_MUSIC = {}

# CLEANING GLOBALS

# for handling the money cleaning
MONEY_PATTERN = r'[-+]?\$?((?:\d+\.\d+)|(?:\d+)|(?:\.\d+))(?:\s*([A-Za-z]+))?'
CENTS_WORDS = ['cents', 'c']
HUNDREDS_WORDS = ['hundred']
THOUSANDS_WORDS = ['thousand', 'k']
MILLIONS_WORDS = ['million', 'm']
BILLIONS_WORDS = ['billion', 'b']

# for handling text cleaning
NOT_WORD_REGEX = r"[^\w\s]"

FEEL_Q = 'Describe how this painting makes you feel.'
FOOD_Q = 'If this painting was a food, what would be?'
MUSIC_Q = 'Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.'

FEEL_IGNORE = [
    'feeling',
    'feelings',
    'feels',
    'felt'
]

FOOD_IGNORE = [
    'foods',
    'food'
]

MUSIC_IGNORE = [
    'sound',
    'sounds',
    'soundtracks',
    'music',
    'song'
]

BASE_IGNORE = set([
    'yourselves', 'as', 'only', 'i', 'no', 'so', 'this', 'www', "shed", 'all', 'you', 'shall', 
    'just', "theyll", 'own', "werent", 'your', "wed", 'for', "hows", "mustnt", "its", "youve", 
    'most', "didnt", 'a', 'over', "havent", "shouldnt", "hes", 'the', 'under', 'each', "thats", 
    'cannot', 'he', 'down', 'because', "arent", 'both', 'what', 'further', "whys", "whens", 
    "theyre", 'or', 'up', 'their', 'how', 'himself', 'me', 'myself', 'him', 'on', 'to', "ill", 
    "hell", 'be', 'k', 'she', 'out', 'them', "were", 'during', "couldnt", 'does', 'her', 'into', 
    'ourselves', 'having', 'by', "hed", "wont", "wheres", 'an', 'before', "im", 'am', 'other', 
    'there', 'but', "isnt", "id", 'otherwise', "shes", 'in', 'few', 'ours', 'of', "weve", 'not', 
    "theres", "youd", 'and', 'ought', 'doing', 'between', 'did', 'than', "theyve", 'herself', 
    "youll", "youre", 'com', 'should', 'been', "shell", 'hence', "well", 'at', 'such', 'theirs', 
    'http', 'again', 'some', 'my', 'else', 'against', "hadnt", "heres", "whats", 'more', 'below', 
    'any', 'after', 'these', 'off', "doesnt", 'can', 'who', "theyd", 'from', 'here', 'why', 'was', 
    'very', 'with', 'therefore', 'like', 'had', 'has', 'nor', 'its', 'those', "wouldnt", 'which', 
    'same', "wasnt", 'itself', 'would', 'above', 'while', 'when', 'however', 'is', 'our', 'where', 
    'yours', 'about', 'if', 'through', 'since', 'ever', 'do', 'themselves', 'r', "cant", 'have', 
    'also', 'hers', 'we', 'until', 'it', 'whom', 'could', 'get', 'yourself', 'too', "dont", "lets", 
    'being', "whos", "ive", 'were', "shant", 'are', 'then', "hasnt", 'once', 'they', 'his', 'that'
    'clock', 'clocks', 'sky', 'night', 'something'
])

"""
Normalizes the columns of the given numpy array to have mean 0 and standard deviation 1.
"""
def normalize(X: np.array) -> np.array:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return (X - mean) / std

"""
Creates similarity vectors for each combination of text response and painting, 
and returns them as a dictionary
"""
def process_text(in_df: pd.DataFrame) -> dict[str, pd.Series]:
    output = {}

    feels = in_df[FEEL_Q]
    food = in_df[FOOD_Q]
    music = in_df[MUSIC_Q]

    qs = [
        (feels, 'feels', FEEL_Q),
        (food, 'food', FOOD_Q),
        (music, 'music', MUSIC_Q)
    ]

    stopwords = set(BASE_IGNORE)

    for painting in ['clock', 'starry', 'lilies']:
        for df, quest, t in qs:
            new_col = []
            no_words = stopwords.copy()

            for word in t.strip('.,!?()[]{}"\'').split():
                no_words.add(word)
            if "feel" in t:
                no_words.update(FEEL_IGNORE)
            elif "food" in t:
                no_words.update(FOOD_IGNORE)
            else:
                no_words.update(MUSIC_IGNORE)

            # very hacky
            weight_d = None
            match f'{painting}-{quest}':
                case 'clock-feels':
                    weight_d = CLOCK_FEEL 
                case 'clock-food':
                    weight_d = CLOCK_FOOD
                case 'clock-music':
                    weight_d = CLOCK_MUSIC
                case 'starry-feels':
                    weight_d = STARRY_FEEL
                case 'starry-food':
                    weight_d = STARRY_FOOD
                case 'starry-music':
                    weight_d = STARRY_MUSIC
                case 'lilies-feels':
                    weight_d = LILIES_FEEL
                case 'lilies-food':
                    weight_d = LILIES_FOOD
                case 'lilies-music':
                    weight_d = LILIES_MUSIC
                case _:
                    raise AttributeError

            for line in df:
                try:
                    val = 0.0
                    for w in line.split():
                        w_clean = re.sub(NOT_WORD_REGEX, "", w.strip().lower())
                        if w_clean in stopwords or w_clean == '':
                            continue
                        if w_clean in weight_d:
                            val += weight_d[w_clean]
                except (ValueError, AttributeError) as e:
                    # this occurs when line is nan
                    val = 0.0

                new_col.append(val)

            new_col_name = f"text_{painting}_{quest}"
            output[new_col_name] = pd.Series(new_col, index=in_df.index)

    return output

"""
Clean the data in the given dataframe, return it as a new dataframe
"""
def clean(data_df):
    # init new df
    out_df = pd.DataFrame({"unique_id": in_df['unique_id']})
    
    # NUMERICAL QUESTIONS

    # emotional intensity
    vals = []
    invalid = []
    i = 0
    for line in in_df.get(["unique_id", 'On a scale of 1–10, how intense is the emotion conveyed by the artwork?']).iterrows():
        try:
            val = int(line[1].iloc[1])
        except ValueError:
            val = 0
            invalid.append(i)

        vals.append(val)
        i += 1

    # replace empty values with the mode
    mode = st.mode(vals)
    for j in invalid:
        vals[j] = mode

    out_df = out_df.assign(emotional_intensity=vals)

    # sombre
    vals = []
    invalid = []
    i = 0
    for line in in_df.get(["unique_id", 'This art piece makes me feel sombre.']).iterrows():
        try:
            val = int(line[1].iloc[1][:1])
        except (ValueError, TypeError) as e:
            val = 0
            invalid.append(i)

        vals.append(val)
        i += 1

    # replace empty values with the mode
    mode = st.mode(vals)
    for j in invalid:
        vals[j] = mode

    out_df = out_df.assign(evokes_sombre=vals)

    # content
    vals = []
    invalid = []
    i = 0
    for line in in_df.get(["unique_id", 'This art piece makes me feel content.']).iterrows():
        try:
            val = int(line[1].iloc[1][:1])
        except (ValueError, TypeError) as e:
            val = 0
            invalid.append(i)

        vals.append(val)
        i += 1

    # replace empty values with the mode
    mode = st.mode(vals)
    for j in invalid:
        vals[j] = mode

    out_df = out_df.assign(evokes_content=vals)

    # calm
    vals = []
    invalid = []
    i = 0
    for line in in_df.get(["unique_id", 'This art piece makes me feel calm.']).iterrows():
        try:
            val = int(line[1].iloc[1][:1])
        except (ValueError, TypeError) as e:
            val = 0
            invalid.append(i)

        vals.append(val)
        i += 1

    # replace empty values with the mode
    mode = st.mode(vals)
    for j in invalid:
        vals[j] = mode

    out_df = out_df.assign(evokes_calm=vals)

    # uneasy
    vals = []
    invalid = []
    i = 0
    for line in in_df.get(["unique_id", 'This art piece makes me feel uneasy.']).iterrows():
        try:
            val = int(line[1].iloc[1][:1])
        except (ValueError, TypeError) as e:
            val = 0
            invalid.append(i)
            
        vals.append(val)
        i += 1

    # replace empty values with the mode
    mode = st.mode(vals)
    for j in invalid:
        vals[j] = mode

    out_df = out_df.assign(evokes_uneasy=vals)

    # number colours
    vals = []
    invalid = []
    i = 0
    for line in in_df.get(["unique_id", 'How many prominent colours do you notice in this painting?']).iterrows():
        try:
            val = int(line[1].iloc[1])
        except ValueError:
            val = 0
            invalid.append(i)

        vals.append(val)
        i += 1

    # replace empty values with the mode
    mode = st.mode(vals)
    for j in invalid:
        vals[j] = mode

    out_df = out_df.assign(prominent_colour_count=vals)

    # number objects
    vals = []
    invalid = []
    i = 0
    for line in in_df.get(["unique_id", 'How many objects caught your eye in the painting?']).iterrows():
        try:
            val = int(line[1].iloc[1])
        except ValueError:
            val = 0
            invalid.append(i)

        vals.append(val)
        i += 1

    # replace empty values with the mode
    mode = st.mode(vals)
    for j in invalid:
        vals[j] = mode

    out_df = out_df.assign(prominent_object_count=vals)

    # CATEGORICAL QUESTIONS

    # what room?
    in_bedroom = []
    in_bathroom = []
    in_office = []
    in_living = []
    in_dining = []
    invalid = []
    for line in in_df.get(["unique_id", "If you could purchase this painting, which room would you put that painting in?"]).iterrows():
        be = 0
        ba = 0
        o = 0
        l = 0
        d = 0
        try:
            val = line[1].iloc[1].lower()
            be = 1 if ('bedroom' in val) else 0
            ba = 1 if ('bathroom' in val) else 0
            o = 1 if ('office' in val) else 0
            l = 1 if ('living' in val) else 0
            d = 1 if ('dining' in val) else 0
        except AttributeError:
            continue

        in_bedroom.append(be)
        in_bathroom.append(ba)
        in_office.append(o)
        in_living.append(l)
        in_dining.append(d)
    
    out_df = out_df.assign(place_bedroom=in_bedroom, place_bathroom=in_bathroom, place_office=in_office, 
                           place_living_room=in_living, place_dining_room=in_dining)

    # view with who?
    friends = []
    family = []
    coworkers = []
    strangers = []
    solo = []
    for line in in_df.get(["unique_id", "If you could view this art in person, who would you want to view it with?"]).iterrows():
        fr = 0
        fa = 0
        c = 0
        st = 0
        so = 0
        try:
            val = line[1].iloc[1].lower()
            fr = 1 if ('friends' in val) else 0
            fa = 1 if ('family' in val) else 0
            c = 1 if ('coworkers' in val) else 0
            st = 1 if ('strangers' in val) else 0
            so = 1 if ('yourself' in val) else 0
        except AttributeError:
            continue
        
        friends.append(fr)
        family.append(fa)
        coworkers.append(c)
        strangers.append(st)
        solo.append(so)

    out_df = out_df.assign(view_friends=friends, view_family=family, 
                           view_coworkers=coworkers, view_strangers=strangers,
                           view_by_yourself=solo)

    # what season?
    fall = []
    winter = []
    spring = []
    summer = []
    for line in in_df.get(["unique_id", "What season does this art piece remind you of?"]).iterrows():
        w = 0
        f = 0
        sp = 0
        su = 0
        try:
            val = line[1].iloc[1].lower()
            w = 1 if ('winter' in val) else 0
            f = 1 if ('fall' in val) else 0
            sp = 1 if ('spring' in val) else 0
            su = 1 if ('summer' in val) else 0
        except AttributeError:
            continue
        
        fall.append(f)
        winter.append(w)
        summer.append(su)
        spring.append(sp)
    
    out_df = out_df.assign(like_fall=fall, like_winter=winter, 
                           like_spring=spring, like_summer=summer)
    
    # handling text-based
    # monetary value
    vals = []
    invalid = []
    i = 0
    for line in in_df.get(["unique_id", 'Painting', "How much (in Canadian dollars) would you be willing to pay for this painting?"]).iterrows():
        try:
            match = re.search(MONEY_PATTERN, line[1].iloc[2].replace(',', '').lower())

            if match:
                number = float(match.group(1))
                word = match.group(2) or ""
                factor = 1

                if word in CENTS_WORDS:
                    factor = 0.01
                elif word in HUNDREDS_WORDS:
                    factor = 100
                elif word in THOUSANDS_WORDS:
                    factor = 1000
                elif word in MILLIONS_WORDS:
                    factor = 1000000
                elif word in BILLIONS_WORDS:
                    factor = 1000000000

                # do not log negative values or 0, set it to log_10(0.001) = -3
                if number <= 0:
                    val = -3
                else:
                    val = math.log10(number * factor)
            else:
                # disgustingly hacky - will treat no match as an invalid val
                raise AttributeError
        except AttributeError:
            val = 0
            invalid.append(i)
            
        vals.append(val)
        i += 1

    avg = st.mean(vals)
    for j in invalid:
        vals[j] = avg

    out_df = out_df.assign(monetary_value=vals)

    # other text questions - create new weight vectors
    out_df = out_df.assign(**process_text(in_df))

    return out_df

"""
Using the given model, make predictions for each row of the input dataframe
"""
def predict(model, data_df):
    return None

""" 
Returns a list of strings, with each string being the name of prediction for 
that row in the given file.
"""
def predict_all(filename):
    predictions = []

    df = pd.read_csv(filename)

    # clean the data
    cleaned = clean(df)

    # init the model
    model = None

    # make a prediction


    return predictions


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <input_file> <output_file OPTIONAL>')
        exit(1)

    infile = sys.argv[1]

    pred_lst = predict_all(infile)

    if len(sys.argv) > 2:
        outfile = sys.argv[2]
        out_df = pd.DataFrame(pred_lst)
        out_df.to_csv(outfile)

    else:
        for p in pred_lst:
            print(pred)