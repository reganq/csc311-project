"""
Prediction script for CSC311 proectt

Regan Quartus, Brenden McFarlane, Elise Corbin
March 26, 2025
"""
import os
import sys
import csv
import statistics as stat
import re
import math
import numpy as np
import pandas as pd

# painting map of painting number to painting name (just an array)
PAINTINGS = [
    "The Persistence of Memory",
    "The Starry Night",
    "The Water Lily Pond"
]

# CLEANED HEADER NAMES
HEADERS = [
    "emotional_intensity", "evokes_sombre", "evokes_content", "evokes_calm",
    "evokes_uneasy", "prominent_colour_count", "prominent_object_count",
    "place_bedroom", "place_bathroom", "place_office", "place_living_room",
    "place_dining_room", "view_friends", "view_family", "view_coworkers",
    "view_strangers", "view_by_yourself", "like_fall", "like_winter", "like_spring",
    "like_summer", "monetary_value", "text_clock_feels", "text_clock_food", 
    "text_clock_music", "text_starry_feels", "text_starry_food", "text_starry_music",
    "text_lilies_feels", "text_lilies_food", "text_lilies_music"
    ]

# SIMILARITY VECTOR WEIGHTS
WEIGHT_DIR = "weights"

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

# IMPORTING PARAMS
PARAMS_DIR = "regression_params"

# MEAN/STD FROM TRAINING SET FOR NORMALIZATION
MEAN = np.array([6.30836454, 2.8102372, 3.38451935, 3.69538077, 2.47690387, 3.9650437,
    3.88139825, 0.40574282, 0.25842697, 0.38077403, 0.54057428, 0.30337079,
    0.56304619, 0.52933833, 0.27091136, 0.24344569, 0.60174782, 0.26716604,
    0.25593009, 0.38701623, 0.4144819,  2.54939535, 0.02978243, 0.01188592,
    0.02571401, 0.02721586, 0.01655445, 0.02434457, 0.03701709, 0.02381223,
    0.02526814])

STD = np.array([2.22314293, 1.38619737, 1.28253808, 1.18441445, 1.39278627, 4.12795065,
    2.84816187, 0.49103522, 0.43776988, 0.48557715, 0.49835101, 0.45971399,
    0.49600925, 0.49913852, 0.44443042, 0.42916184, 0.48953793, 0.44247977,
    0.43638272, 0.48706742, 0.49263237, 2.33375606, 0.05353837, 0.01545048,
    0.03016835, 0.02773692, 0.02550723, 0.0244587,  0.04127373, 0.04207384,
    0.02512345])

"""
Logistic Regression class. Comprised of:
    - weights: numpy matrix of weights, of dimension D_in x D_out
    - intercept: numpy vector of weights, of dimension D_out

"""
class LogisticRegression:
    weights = None
    intercept = None

    """
    Performs inference using the model on the given input.
        - input_data: numpy array of shape (N, D0) where D0 is the dimension of inputs
    """
    def predict(self, input_data):
        N = input_data.shape[0]

        arr = [self.intercept.copy() for _ in range(N)]

        tmp = np.dot(input_data, self.weights) + np.stack(arr)

        return np.argmax(tmp, axis=1)

"""
Normalizes the columns of the given numpy array to have mean 0 and standard deviation 1.
"""
def normalize(X: np.array) -> np.array:
    # TODO: Determine if we should be using mean/std from the training data or if using test vals is fine
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return (X - MEAN) / STD

"""
Load all 9 weight dictionaries from the input directory.
"""
def load_weights(weight_dir):
    for paint in ["clocks", "starry", "lilies"]:
        for quest in ["feels", "food", "music"]:
            d = {}
            with open(f"{weight_dir}/{paint}-{quest}.txt", 'r') as fp:
                for line in fp:
                    sline = line.split()
                    word = sline[0]
                    weight = float(sline[1])
                    d[word] = weight
      
            # very hacky
            if f'{paint}-{quest}' == 'clocks-feels':
                CLOCK_FEEL.update(d)
            elif f'{paint}-{quest}' == 'clocks-food':
                CLOCK_FOOD.update(d)
            elif f'{paint}-{quest}' == 'clocks-music':
                CLOCK_MUSIC.update(d)
            elif f'{paint}-{quest}' == 'starry-feels':
                STARRY_FEEL.update(d)
            elif f'{paint}-{quest}' == 'starry-food':
                STARRY_FOOD.update(d)
            elif f'{paint}-{quest}' == 'starry-music':
                STARRY_MUSIC.update(d)
            elif f'{paint}-{quest}' == 'lilies-feels':
                LILIES_FEEL.update(d)
            elif f'{paint}-{quest}' == 'lilies-food':
                LILIES_FOOD.update(d)
            elif f'{paint}-{quest}' == 'lilies-music':
                LILIES_MUSIC.update(d)
            else:
                raise AttributeError


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
def clean(in_df):
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
    mode = stat.mode(vals)
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
    mode = stat.mode(vals)
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
    mode = stat.mode(vals)
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
    mode = stat.mode(vals)
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
    mode = stat.mode(vals)
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
    mode = stat.mode(vals)
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
    mode = stat.mode(vals)
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
        try:
            val = line[1].iloc[1].lower()
            be = 1 if ('bedroom' in val) else 0
            ba = 1 if ('bathroom' in val) else 0
            o = 1 if ('office' in val) else 0
            l = 1 if ('living' in val) else 0
            d = 1 if ('dining' in val) else 0
        except AttributeError:
            be = 0
            ba = 0
            o = 0
            l = 0
            d = 0

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
        try:
            val = line[1].iloc[1].lower()
            fr = 1 if ('friends' in val) else 0
            fa = 1 if ('family' in val) else 0
            c = 1 if ('coworkers' in val) else 0
            st = 1 if ('strangers' in val) else 0
            so = 1 if ('yourself' in val) else 0
        except AttributeError:
            fr = 0
            fa = 0
            c = 0
            st = 0
            so = 0
        
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
        try:
            val = line[1].iloc[1].lower()
            w = 1 if ('winter' in val) else 0
            f = 1 if ('fall' in val) else 0
            sp = 1 if ('spring' in val) else 0
            su = 1 if ('summer' in val) else 0
        except AttributeError:
            w = 0
            f = 0
            sp = 0
            su = 0
        
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
    for line in in_df.get(["unique_id", "How much (in Canadian dollars) would you be willing to pay for this painting?"]).iterrows():
        try:
            match = re.search(MONEY_PATTERN, line[1].iloc[1].replace(',', '').lower())

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

    avg = stat.mean(vals)
    for j in invalid:
        vals[j] = avg

    out_df = out_df.assign(monetary_value=vals)

    # other text questions - create new weight vectors
    out_df = out_df.assign(**process_text(in_df))

    return out_df

"""
Initializes a regression model by reading it's weights from the given directory of params
"""
def init_regression(directory):
    lreg = LogisticRegression()
    
    # load each file from the given directory
    lreg.weights = np.load(os.path.join(directory, f'weights.npy')).T
    lreg.intercept = np.load(os.path.join(directory, f'bias.npy'))

    return lreg

"""
Using the given model, make predictions for each row of the input dataframe
"""
def predict(model, data_df):
    preds = []

    # normalize the data and convert it to NP array
    X = np.array(data_df[HEADERS])
    X = normalize(X)

    # model specific - determine predictions
    pred_vals = model.predict(X)

    # convert predictions into text strings
    for p in np.nditer(pred_vals):
        preds.append(PAINTINGS[p])

    return preds

""" 
Returns a list of strings, with each string being the name of prediction for 
that row in the given file.
"""
def predict_all(filename):
    df = pd.read_csv(filename)

    # load the weights
    load_weights(WEIGHT_DIR)

    # clean the data
    cleaned = clean(df)
    cleaned.to_csv("tmp.csv", index=False)

    # init the model
    model = init_regression(PARAMS_DIR)

    # make a prediction
    predictions = predict(model, cleaned)

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

    # compute and print out accuracy
    input_d = pd.read_csv(infile)
    n_correct = 0
    n = 0
    for row in input_d['Painting']:
        if (row == pred_lst[n]):
            n_correct += 1
        n += 1

    print(f'Number of test examples: {n}')
    print(f'Test accuracy: {n_correct / n}')
