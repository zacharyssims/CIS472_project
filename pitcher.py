import pandas as pd
#Verlander:
##DET(2008-2017) HOU(2017-2019)
def verlander_home(verlander):
    if verlander['home_team'] == 'DET':
        if verlander['game_year'] in [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]:
                return 1
        else: return 0 
    elif verlander['home_team'] == 'HOU':
        if verlander['away_team'] == 'DET':
            if verlander['game_year'] == 2017:
                return 0
            else: return 0 
        elif verlander['game_year'] in [2017,2018,2019]:
            return 1    
        else: return 0   
    else: return 0 
#Lester:
## BOS(2008-2014) OAK(2014) CHC(2015-2019)
def lester_home(lester):
    if lester['home_team'] == 'BOS':
        if lester['game_year'] in [2008,2009,2010,2011,2012,2013,2014]:
            return 1
        else: return 0
    elif lester['home_team'] == 'CHC':
        if lester['game_year'] in [2015,2016,2017,2018,2019]:
            return 1
        else:
            return 0
    elif lester['home_team'] == 'OAK':
        if lester['away_team'] == 'BOS':
            if lester['game_year'] == 2014:
                return 0
            else:
                return 0
        elif lester['game_year'] == 2014:
            return 1
        else: return 0
    else:
        return 0
#Scherzer
## ARI(2008-2009) DET(2010-2014) WAS(2015-2019)
def scherzer_home(scherzer):
    if scherzer['home_team'] == 'ARI':
        if scherzer['game_year'] in [2008,2009]:
            return 1
        else: return 0
    elif scherzer['home_team'] == 'DET':
        if scherzer['game_year'] in [2010,2011,2012,2013,2014]:
            return 1
        else: return 0
    elif scherzer['home_team'] == 'WAS':
        if scherzer['game_year'] in [2015,2016,2017,2018,2019]:
            return 1 
        else: return 0
    else: return 0
#Hamels:
##PHI(2008-2015) TEX(2015-2018) CHC(2018-2019) 
def hamels_home(hamels):
    if hamels['home_team'] == 'PHI':
        if hamels['game_year'] in [2008,2009,2010,2011,2012,2013,2014,2015]:
            return 1
        else: return 0
    elif hamels['home_team'] == 'TEX':
        if hamels['game_year'] in [2015,2016,2017,2018]:
            return 1
        else: return 0
    elif hamels['home_team'] == 'CHC':
        if hamels['game_year'] in [2018,2019]:
            return 1
        else: return 0
    else: return 0

def clean_data(pitcher):
    pitcher['on_1b'] = pitcher['on_1b'].apply(lambda row: 1 if row == row else 0)
    pitcher['on_2b'] = pitcher['on_2b'].apply(lambda row: 1 if row == row else 0)
    pitcher['on_3b'] = pitcher['on_3b'].apply(lambda row: 1 if row == row else 0)
    pitcher['stand'] = pitcher['stand'].apply(lambda row: 1 if row == 'L' else 0)
    pitcher.rename(columns={'stand':'stand_L'},inplace=True)
    if pitcher['player_name'][0] == 'Justin Verlander':
        pitcher['home_team'] = pitcher.apply(lambda row: verlander_home(row),axis=1)
    if pitcher['player_name'][0] == 'Jon Lester':
        pitcher['home_team'] = pitcher.apply(lambda row: lester_home(row),axis=1)
    if pitcher['player_name'][0] == 'Cole Hamels':
        pitcher['home_team'] = pitcher.apply(lambda row: hamels_home(row),axis=1)
    if pitcher['player_name'][0] == 'Max Scherzer':
        pitcher['home_team'] = pitcher.apply(lambda row: scherzer_home(row),axis=1)  
    pitcher_cols = ['game_pk','pitch_type','pfx_x','pfx_z','description','release_speed',
                    'plate_x','plate_z','stand_L', 'home_team', 'balls','strikes',
                    'on_3b', 'on_2b', 'on_1b', 'outs_when_up']
    pitcher = pitcher[pitcher_cols]
    balls_ohe = pd.get_dummies(pitcher['balls'],prefix='balls')
    strikes_ohe = pd.get_dummies(pitcher['strikes'],prefix='strikes')
    outs_ohe = pd.get_dummies(pitcher['outs_when_up'],prefix='outs')
    des_ohe = pd.get_dummies(pitcher['description'])
    pitcher.drop(['balls','strikes','outs_when_up','description'],axis=1,inplace=True)
    pitcher =  pitcher.join(balls_ohe)
    pitcher = pitcher.join(strikes_ohe)
    pitcher = pitcher.join(outs_ohe)
    pitcher = pitcher.join(des_ohe)
    return pitcher

#transform data into a list of games 
def get_games(pitcher):
    game_id = pitcher['game_pk']
    pitcher = pitcher.values
    games = []
    game = []
    for index in reversed(range(len(game_id))):
        game.append(pitcher[index])
        if index != (len(game_id) - 1) and game_id[index] != game_id[index + 1]:
            games.append(game)
            game = []
    return games  


#transform data into representations for model, i.e. (PREV_PITCHES,PREV_PTYPES,PRE_PITCH,PTYPE) where:
#PREV_PITCHES is a list of 5 vectors, each representing the last five pitches in terms of velocity, plate location,
#horizontal and vertical movement, and outcome.
#PREV_TYPES are the previous 5 pitch types
#PRE_PITCH is a  one_hot vector describing the pre-pitch game state for the pitch to predict, in terms of number of outs,
#BALLS, and strikes, runners on base, if the pitcher is home or not, and whether the batter is left handed
#PTYPE is the target pitch type for prediction 
def get_reps(pitcher_games):
    reps = []
    for game in pitcher_games:
        ptypes = [pitch[1] for pitch in game]
        prev_pitch_cont = [pitch[2:7].tolist() for pitch in game]
        prev_pitch_disc = [pitch[22:].tolist() for pitch in game]
        pitches = zip(prev_pitch_cont,prev_pitch_disc)
        prev_pitches = [cont + disc for cont,disc in pitches]
        pre_pitch = [pitch[7:22].tolist() for pitch in game]
        game_len = len(game)
        for i in range(game_len):
            if i < (game_len - 1) - 6:
                rep = (prev_pitches[i:i+5],ptypes[i:i+5],pre_pitch[i+5],ptypes[i+5])
                reps.append(rep)
    return reps 

#drop any representation that contains a NaN value
def drop_nas(reps):
    good_reps = []
    for rep in reps:
        is_na = False
        prev_pitches,prev_types,pre_pitch,ptype = rep
        if ptype != ptype:
            is_na = True
        else:
            for pitch in prev_pitches:
                for stat in pitch:
                    if stat != stat:
                        is_na = True
            for pitch in prev_types:
                if pitch != pitch:
                    is_na = True
        if not is_na:
            good_reps.append(rep)
            is_na = False
    return good_reps  

#drop half the fastballs in the set of representations, this creates an even distribution of ptypes in the training set
def drop_ff(reps):
    good_reps = []
    i = 0
    for rep in reps:
        prev_pitches,prev_types,pre_pitch,ptype = rep
        if ptype == 'FF':
            i += 1
            if i%4 == 0:
                good_reps.append(rep)
        else:
            good_reps.append(rep)
    return good_reps

#count the number of fourseam-fastballs in the set of representations
def num_ff(reps):
    i = 0
    for rep in reps:
            prev_pitches,prev_types,pre_pitch,ptype = rep
            if ptype == 'FF':
                i += 1
    return i

def drop_pitches(reps):
    good_reps = []
    for rep in reps:
        prev_pitches,prev_types,pre_pitch,ptype = rep
        if ptype in ['CU','FF','SL','CH']:
                good_reps.append(rep)
    return good_reps 