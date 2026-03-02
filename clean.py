# clean the data and save it
#
# Feb 2026
import sys
import pandas as pd
import numpy as np
import statistics as st
import re

# training-validation split. For now, use 50% train, 20% val, 30% test so
TRAIN_NUM = 5
TRAIN_DENOM = 7

# for handling the money cleaning
MONEY_PATTERN = r'[-+]?\$?((?:\d+\.\d+)|(?:\d+)|(?:\.\d+))(?:\s*([A-Za-z]+))?'
CENTS_WORDS = ['cents', 'c']
HUNDREDS_WORDS = ['hundred']
THOUSANDS_WORDS = ['thousand', 'k']
MILLIONS_WORDS = ['million', 'm']
BILLIONS_WORDS = ['billion', 'b']

# for computing text vectors
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

CLOCK_FEEL = {}
CLOCK_FOOD = {}
CLOCK_MUSIC = {}
LILIES_FEEL = {}
LILIES_FOOD = {}
LILIES_MUSIC = {}
STARRY_FEEL = {}
STARRY_FOOD = {}
STARRY_MUSIC = {}

# load the weight dictionaries from the given directory
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

# compute the counts       
def counts(in_df, title) -> tuple[dict[str, int], int]:
    stopwords = set(BASE_IGNORE)
    for word in title.strip('.,!?()[]{}"\'').split():
        stopwords.add(word)

    if "feel" in title:
        stopwords.update(FEEL_IGNORE)
    elif "food" in title:
        stopwords.update(FOOD_IGNORE)
    else:
        stopwords.update(MUSIC_IGNORE)

    counts = dict()
    total = 0
    for line in in_df.dropna():
        for w in line.split():
            w_clean = re.sub(r"[^\w\s]", "", w.strip().lower())
            if w_clean in stopwords or w_clean == '':
                continue

            if w_clean not in counts:
                counts[w_clean] = 0
            counts[w_clean] += 1
            total += 1

    return counts, total

# compute the per-painting, per-question weights using the given dataframe
def compute_weights(in_df: pd.DataFrame):
    clocks = in_df[in_df['Painting'] == 'The Persistence of Memory']
    starry = in_df[in_df['Painting'] == 'The Starry Night']
    lilies = in_df[in_df['Painting'] == 'The Water Lily Pond']

    cols = [
        ("Describe how this painting makes you feel.", "feels"),
        ("If this painting was a food, what would be?", "food"),
        ("Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.", "music")]

    # compute counts for each painting and each question
    for df, paint in [(clocks, "clocks"), (starry, "starry"), (lilies, "lilies")]:
        for col, quest in cols:
            col_df = df[col]
            d, total = counts(col_df, col)
            for k in d:
                d[k] /= total

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

def process_text(in_df: pd.DataFrame) -> dict[str, pd.Series]:
    output = {}

    feels = in_df['Describe how this painting makes you feel.']
    food = in_df['If this painting was a food, what would be?']
    music = in_df['Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.']
  
    qs = [
        (feels, 'feels', 'Describe how this painting makes you feel.'),
        (food, 'food', 'If this painting was a food, what would be?'),
        (music, 'music', 'Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.')
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
            dir = None
            match f'{painting}-{quest}':
                case 'clock-feels':
                    dir = CLOCK_FEEL 
                case 'clock-food':
                    dir = CLOCK_FOOD
                case 'clock-music':
                    dir = CLOCK_MUSIC
                case 'starry-feels':
                    dir = STARRY_FEEL
                case 'starry-food':
                    dir = STARRY_FOOD
                case 'starry-music':
                    dir = STARRY_MUSIC
                case 'lilies-feels':
                    dir = LILIES_FEEL
                case 'lilies-food':
                    dir = LILIES_FOOD
                case 'lilies-music':
                    dir = LILIES_MUSIC
                case _:
                    raise AttributeError

            for line in df:
                try:
                    val = 0.0
                    for w in line.split():
                        w_clean = re.sub(r"[^\w\s]", "", w.strip().lower())
                        if w_clean in stopwords or w_clean == '':
                            continue
                        if w_clean in dir:
                            val += dir[w_clean]
                except (ValueError, AttributeError) as e:
                    # this occurs when line is nan
                    val = 0.0

                new_col.append(val)

            new_col_name = f"text_{painting}_{quest}"
            output[new_col_name] = pd.Series(new_col)

    return output


# clean the given dataframe by applying a number of rules
def clean(in_df: pd.DataFrame) -> pd.DataFrame:
    out_columns = [
        'unique_id',
        'painting'
        'emotional_intensity',
        'evokes_sombre',
        'evokes_content',
        'evokes_calm',
        'evokes_unease',
        'prominent_colour_count',
        'prominent_object_count',
        'place_bedroom',
        'place_office',
        'place_living_room',
        'place_dining_room',
        'view_friends',
        'view_family',
        'view_coworkers',
        'view_by_yourself',
        'like_fall',
        'like_winter',
        'like_spring',
        'like_summer',
        'monetary_value',
        'text_clock_feel',
        'text_clock_food',
        'text_clock_music',
        'text_starry_feel',
        'text_starry_food',
        'text_starry_music',
        'text_lilies_feel',
        'text_lilies_food',
        'text_lilies_music',
    ]

    # compute the IDs - seed the random choice so that we get a repeatable outcome
    rng = np.random.default_rng(seed=123456789)
    # divide the shape by 3 because there are 3 paintings
    sz = in_df.shape[0] // 3
    before = np.array(in_df['unique_id'].unique())
    samples = list(rng.choice(before, (sz * TRAIN_NUM) // TRAIN_DENOM, replace=False))
    is_train = in_df['unique_id'].isin(samples)

    # init new df
    out_df = pd.DataFrame({"unique_id": in_df['unique_id'], 'painting': in_df['Painting'], 
                           'is_train': is_train})
    
    # handle numerical values - if not present, should we take mean or mode?

    # emotional intensity
    vals = []
    invalid = []
    for line in in_df.get(["unique_id", 'On a scale of 1–10, how intense is the emotion conveyed by the artwork?']).iterrows():
        try:
            val = int(line[1].iloc[1])
        except ValueError:
            val = 0
            invalid.append((line[1].iloc[0], line[1].iloc[1]))
        vals.append(val)
    print(f'For emotional_intensity, valid are {len(vals) - len(invalid)} / {len(vals)}')
    print('invalid are :' + ' '.join([f'({i[0]}, {i[1]}),' for i in invalid]))
    print(f'mean: {st.mean(vals)}, mode: {st.mode(vals)} median: {st.median(vals)}')
    out_df = out_df.assign(emotional_intensity=vals)

    # sombre
    vals = []
    invalid = []
    for line in in_df.get(["unique_id", 'This art piece makes me feel sombre.']).iterrows():
        try:
            val = int(line[1].iloc[1][:1])
        except (ValueError, TypeError) as e:
            val = 0
            invalid.append((line[1].iloc[0], line[1].iloc[1]))
        vals.append(val)
    print(f'For evokes_sombre, valid are {len(vals) - len(invalid)} / {len(vals)}')
    print('invalid are :' + ' '.join([f'({i[0]}, {i[1]}),' for i in invalid]))
    print(f'mean: {st.mean(vals)}, mode: {st.mode(vals)} median: {st.median(vals)}')
    out_df = out_df.assign(evokes_sombre=vals)

    # content
    vals = []
    invalid = []
    for line in in_df.get(["unique_id", 'This art piece makes me feel content.']).iterrows():
        try:
            val = int(line[1].iloc[1][:1])
        except (ValueError, TypeError) as e:
            val = 0
            invalid.append((line[1].iloc[0], line[1].iloc[1]))
        vals.append(val)
    print(f'For evokes_content, valid are {len(vals) - len(invalid)} / {len(vals)}')
    print('invalid are :' + ' '.join([f'({i[0]}, {i[1]}),' for i in invalid]))
    print(f'mean: {st.mean(vals)}, mode: {st.mode(vals)} median: {st.median(vals)}')
    out_df = out_df.assign(evokes_content=vals)

    # calm
    vals = []
    invalid = []
    for line in in_df.get(["unique_id", 'This art piece makes me feel calm.']).iterrows():
        try:
            val = int(line[1].iloc[1][:1])
        except (ValueError, TypeError) as e:
            val = 0
            invalid.append((line[1].iloc[0], line[1].iloc[1]))
        vals.append(val)
    print(f'For evokes_calm valid are {len(vals) - len(invalid)} / {len(vals)}')
    print('invalid are :' + ' '.join([f'({i[0]}, {i[1]}),' for i in invalid]))
    print(f'mean: {st.mean(vals)}, mode: {st.mode(vals)} median: {st.median(vals)}')
    out_df = out_df.assign(evokes_calm=vals)

    # uneasy
    vals = []
    invalid = []
    for line in in_df.get(["unique_id", 'This art piece makes me feel uneasy.']).iterrows():
        try:
            val = int(line[1].iloc[1][:1])
        except (ValueError, TypeError) as e:
            val = 0
            invalid.append((line[1].iloc[0], line[1].iloc[1]))
        vals.append(val)
    print(f'For evokes_uneasy, valid are {len(vals) - len(invalid)} / {len(vals)}')
    print('invalid are :' + ' '.join([f'({i[0]}, {i[1]}),' for i in invalid]))
    print(f'mean: {st.mean(vals)}, mode: {st.mode(vals)} median: {st.median(vals)}')
    out_df = out_df.assign(evokes_uneasy=vals)

    # number colours
    vals = []
    invalid = []
    for line in in_df.get(["unique_id", 'How many prominent colours do you notice in this painting?']).iterrows():
        try:
            val = int(line[1].iloc[1])
        except ValueError:
            val = 0
            invalid.append((line[1].iloc[0], line[1].iloc[1]))
        vals.append(val)
    print(f'For prominent_colour_count, valid are {len(vals) - len(invalid)} / {len(vals)}')
    print('invalid are :' + ' '.join([f'({i[0]}, {i[1]}),' for i in invalid]))
    print(f'mean: {st.mean(vals)}, mode: {st.mode(vals)} median: {st.median(vals)}')
    out_df = out_df.assign(prominent_colour_count=vals)

    # number objects
    vals = []
    invalid = []
    for line in in_df.get(["unique_id", 'How many objects caught your eye in the painting?']).iterrows():
        try:
            val = int(line[1].iloc[1])
        except ValueError:
            val = 0
            invalid.append((line[1].iloc[0], line[1].iloc[1]))
        vals.append(val)
    print(f'For prominent_object_count, valid are {len(vals) - len(invalid)} / {len(vals)}')
    print('invalid are :' + ' '.join([f'({i[0]}, {i[1]}),' for i in invalid]))
    print(f'mean: {st.mean(vals)}, mode: {st.mode(vals)} median: {st.median(vals)}')
    out_df = out_df.assign(prominent_object_count=vals)

    # handle categorical values

    # what room?
    in_bedroom = []
    in_office = []
    in_living = []
    in_dining = []
    invalid = []
    for line in in_df.get(["unique_id", "If you could purchase this painting, which room would you put that painting in?"]).iterrows():
        b = 0
        o = 0
        l = 0
        d = 0
        try:
            val = line[1].iloc[1].lower()
            b = 1 if ('bathroom' in val) else 0
            o = 1 if ('office' in val) else 0
            l = 1 if ('living' in val) else 0
            d = 1 if ('dining' in val) else 0
        except AttributeError:
            invalid.append((line[1].iloc[0], line[1].iloc[1]))
        
        in_bedroom.append(b)
        in_office.append(o)
        in_living.append(l)
        in_dining.append(d)
    
    print(f'For room, valid are {len(in_bedroom) - len(invalid)} / {len(in_bedroom)}')
    print('invalid are :' + ' '.join([f'({i[0]}, {i[1]}),' for i in invalid]))
    out_df = out_df.assign(place_bedroom=in_bedroom, place_office=in_office, 
                           place_living_room=in_living, place_dining_room=in_dining)

    # view with who?
    friends = []
    family = []
    coworkers = []
    solo = []
    invalid = []
    for line in in_df.get(["unique_id", "If you could view this art in person, who would you want to view it with?"]).iterrows():
        fr = 0
        fa = 0
        c = 0
        s = 0
        try:
            val = line[1].iloc[1].lower()
            fr = 1 if ('friends' in val) else 0
            fa = 1 if ('family' in val) else 0
            c = 1 if ('coworkers' in val) else 0
            s = 1 if ('yourself' in val) else 0
        except AttributeError:
            invalid.append((line[1].iloc[0], line[1].iloc[1]))
        
        friends.append(fr)
        family.append(fa)
        coworkers.append(c)
        solo.append(s)
    
    print(f'For viewing with, valid are {len(friends) - len(invalid)} / {len(friends)}')
    print('invalid are :' + ' '.join([f'({i[0]}, {i[1]}),' for i in invalid]))
    out_df = out_df.assign(view_friends=friends, view_family=family, 
                           view_coworkers=coworkers, view_by_yourself=solo)

    # what season?
    fall = []
    winter = []
    spring = []
    summer = []
    invalid = []
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
            invalid.append((line[1].iloc[0], line[1].iloc[1]))
        
        fall.append(f)
        winter.append(w)
        summer.append(su)
        spring.append(sp)
    
    print(f'For season, valid are {len(friends) - len(invalid)} / {len(friends)}')
    print('invalid are :' + ' '.join([f'({i[0]}, {i[1]}),' for i in invalid]))
    out_df = out_df.assign(like_fall=fall, like_winter=winter, 
                           like_spring=spring, like_summer=summer)
    
    # handling text-based
    # monetary value
    vals = []
    invalid = []
    nomatch = []
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

                val = number * factor
            else:
                val = 0
                nomatch.append((line[1].iloc[0], line[1].iloc[1]))
        except AttributeError:
            val = 0
            invalid.append((line[1].iloc[0], line[1].iloc[1]))
        vals.append(val)
    print(f'For monetary_value, valid are {len(vals) - len(invalid) - len(nomatch)} / {len(vals)}')
    print('non matching are :' + ' '.join([f'({i[0]}, {i[1]}),' for i in nomatch]))
    print('invalid are :' + ' '.join([f'({i[0]}, {i[1]}),' for i in invalid]))
    print(f'mean: {st.mean(vals)}, mode: {st.mode(vals)} median: {st.median(vals)}')
    out_df = out_df.assign(monetary_value=vals)

    # other text questions - create new weight vectors
    out_df = out_df.assign(**process_text(in_df))

    return out_df

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'usage: {sys.argv[0]} <data-path> <out-path> <ids-path OPTIONAL> <weights-path OPTIONAL>')
        exit(1)

    dpath = sys.argv[1]
    df = pd.read_csv(dpath)

    if len(sys.argv) > 3:
        # use given set of indices - otherwise cleans the whole thing
        ipath = sys.argv[3]
        ids = np.loadtxt(ipath, dtype=[("id", "i4")])
        df = df[df["unique_id"].isin(ids["id"])]

    if len(sys.argv) > 4:
        # load the weight dictionaries from the given file
        load_weights(sys.argv[4])
    else:
        # compute the weights from the dataframe
        compute_weights(df)

    clean_df = clean(df)

    opath = sys.argv[2]
    clean_df.to_csv(opath, index=False)
