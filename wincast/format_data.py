import pandas as pd


def off_win(row, games_df):
    score = games_df.iloc[row.loc['gid'] - 1]

    if row['off'] == score['h']:

        if score.ptsh > score.ptsv:
            return 1
        else:
            return 0


    elif row['off'] == score['v']:

        if score.ptsv > score.ptsh:
            return 1

        else:
            return 0

    else:
        raise Exception('Teams do not match: %s vs. %s; %s vs. %s' % (
            row['off'], row['def'], score['v'], score['h']))


def get_ou(row, games_df):
    game = games_df.iloc[row.loc['gid'] - 1]

    return game.ou


def get_off_pt_spread(row, games_df):
    games = games_df.iloc[row.loc['gid'] - 1]

    if games.v == row.off:
        return games.sprv
    elif games.v == row['def']:
        return -int(games.sprv)
        



def main():
    plays_df = pd.read_csv('./data/raw/PLAY.csv')
    games_df = pd.read_csv('./data/raw/GAME.csv')

    plays_df = plays_df.loc[:, [
        'gid',
        'off',
        'def',
        'qtr',
        'min',
        'sec',
        'ptso',
        'ptsd',
        'timo',
        'timd',
        'dwn',
        'ytg',
        'yfog'
    ]]

    plays_df['ou'] = plays_df.apply(lambda row: get_ou(row, games_df), axis=1)
    plays_df['pts_s'] = plays_df.apply(
        lambda row: get_off_pt_spread(row, games_df), axis=1)
    plays_df['y'] = plays_df.apply(lambda row: off_win(row, games_df), axis=1)
    plays_df.to_csv('./data/Xy.csv')


if __name__ == '__main__':
    main()
