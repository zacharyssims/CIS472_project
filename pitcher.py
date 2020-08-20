import pandas as pd
i
hello = "hello"
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
    pitcher_cols = ['batter','pitch_type','pfx_x','pfx_z','description','release_speed',
                    'plate_x','plate_z','stand_L', 'home_team', 'balls','strikes',
                    'on_3b', 'on_2b', 'on_1b', 'outs_when_up']
    pitcher = pitcher[pitcher_cols]
    balls_ohe = pd.get_dummies(pitcher['balls'],prefix='balls')
    strikes_ohe = pd.get_dummies(pitcher['strikes'],prefix='strikes')
    outs_ohe = pd.get_dummies(pitcher['outs_when_up'],prefix='outs')
    des_ohe = pd.get_dummies(pitcher['description'])
    pitch_ohe = pd.get_dummies(pitcher['pitch_type'])
    pitcher.drop(['balls','strikes','outs_when_up','description'],axis=1,inplace=True)
    pitcher =  pitcher.join(balls_ohe)
    pitcher = pitcher.join(strikes_ohe)
    pitcher = pitcher.join(outs_ohe)
    pitcher = pitcher.join(des_ohe)
    pitcher = pitcher.join(pitch_ohe)
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

def get_abs(pitcher):
    batter_id = pitcher['batter']
    pitcher = pitcher.values
    ABs = []
    AB = []
    for index in reversed(range(len(batter_id))):
        AB.append(pitcher[index])
        if index != (len(batter_id) - 1) and batter_id[index] != batter_id[index + 1]:
            ABs.append(AB)
            AB = []
    return ABs  


#transform data into representations for model, i.e. (PREV_PITCHES,PREV_PTYPES,PRE_PITCH,PTYPE) where:
#PREV_PITCHES is a list of 5 vectors, each representing the last five pitches in terms of velocity, plate location,
#horizontal and vertical movement, and outcome.
#PREV_TYPES are the previous 5 pitch types
#PRE_PITCH is a  one_hot vector describing the pre-pitch game state for the pitch to predict, in terms of number of outs,
#BALLS, and strikes, runners on base, if the pitcher is home or not, and whether the batter is left handed
#PTYPE is the target pitch type for prediction 
'''def get_reps(pitcher_games):
    reps = []
    for game in pitcher_games:
        ptypes = [pitch[1] for pitch in game]
        prev_pitch_cont = [pitch[2:7].tolist() for pitch in game]
        prev_pitch_disc = [pitch[22:38].tolist() for pitch in game]
        ptypes_ = [pitch[38:42].tolist() for pitch in game]
        pitches = zip(prev_pitch_cont,prev_pitch_disc)
        prevs = [cont + disc for cont,disc in pitches]
        pitches_ = zip(prevs,ptypes_)
        prev_pitches = [prev + ptype for prev,ptype in pitches_]
        pre_pitch = [pitch[7:22].tolist() for pitch in game]
        game_len = len(game)
        for i in range(game_len):
            if i < (game_len - 1) - 6:
                rep = (prev_pitches[i:i+5],ptypes[i:i+5],pre_pitch[i+5],ptypes[i+5])
                reps.append(rep)
    return reps '''
def get_reps(pitcher_ABs):
    reps = []
    for AB in pitcher_ABs:
        prev_0 = [0]*25
        ptypes = []
        for pitch in AB:
            if pitch[1] != 'FF':
                ptypes.append('NF') 
            else:
                ptypes.append('FF')

        prev_pitch_cont = [pitch[2:7].tolist() for pitch in AB]
        prev_pitch_disc = [pitch[22:38].tolist() for pitch in AB]
        ptypes_ = [pitch[38:42].tolist() for pitch in AB]
        pitches = zip(prev_pitch_cont,prev_pitch_disc)
        ###pitches = zip(prev_pitch_cont,ptypes_)
        prevs = [cont + disc for cont,disc in pitches]
        #prevs = prev_pitch_cont
        pitches_ = zip(prevs,ptypes_)
        ###prev_pitches = [prev + ptype for prev,ptype in pitches]
        prev_pitches = [prev + ptype for prev,ptype in pitches_]
        pre_pitch = [pitch[7:22].tolist() for pitch in AB]
        #prev_p = zip(prev_pitches_,pre_pitch)
        #prev_pitches = [prev + prepitch for prev,prepitch in prev_p]
        AB_len = len(AB)
        if AB_len <= 6:
            prevs = [prev_pitches] + ([prev_0]*(6-AB_len))
            states = [[0]*15]+[pre_pitch[1:]] + ([[0]*15]*(6-AB_len))
            pitches = ['NF'] + [ptypes[1:]] + (['NF']*(6-AB_len))
        else:
            prevs = [prev_pitches[:7]]
            states = [[0]*15]+[pre_pitch[1:7]]
            pitches = ['NF'] + [ptypes[1:7]]
        #reps.append(prevs,states,pitches)

        for i in range(AB_len):
            if i == 0:
                prevs = [prev_0]*3
                prev_ptypes = ['NA']*3
            elif i == 1:
                prevs = [prev_0]*2 + [prev_pitches[i-1]]
                prev_ptypes =  ['NA']*2 + [ptypes[i-1]]
            elif i == 2:
                prevs = [prev_0] + prev_pitches[i-2:i]
                prev_ptypes =  ['NA'] + [ptypes[i-2:i]]
            else:
                prevs = prev_pitches[i-3:i]
                prev_ptypes =  ptypes[i-3:i]
            reps.append((prevs,prev_ptypes,pre_pitch[i],ptypes[i]))
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


def get_batches(train,batch_size):   
    out = []
    i = 0
    batch_ptypes = []
    batch_pre_pitch = []
    batch_prev_pitches = []
    for rep in train:
        prev_pitches,prev_types,pre_pitch,ptype = rep
        if i % batch_size == 0:
            out.append([batch_prev_pitches,batch_pre_pitch,batch_ptypes])
            batch_ptypes = [ptype]
            batch_pre_pitch = [pre_pitch]
            batch_prev_pitches = [prev_pitches]
        else:
            batch_ptypes.append(ptype)
            batch_pre_pitch.append(pre_pitch)
            batch_prev_pitches.append(prev_pitches)
        i += 1
    return out[1:]
