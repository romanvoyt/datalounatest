import pandas as pd

def preprocess_data(train_df, players_df, test_df):
    p1_df = players_df.iloc[:, 1:25]
    p2_df = players_df.iloc[:, 26:50]
    p3_df = players_df.iloc[:, 51:75]
    p4_df = players_df.iloc[:, 76:100]
    p5_df = players_df.iloc[:, 101:125]

    columns = ['team_avg_total_kills', 'team_avg_headshots', 'team_avg_total_deaths', 'team_avg_kd_ratio',
               'team_avg_damage_per_round', 'team_avg_grenade_damage_per_round', 'team_avg_maps_played',
               'team_avg_rounds_played', 'team_avg_kills_per_round', 'team_avg_assists_per_round',
               'team_avg_deaths_per_round', 'team_avg_saved_by_teammates_per_round',
               'team_avg_saved_teammates_per_round', 'team_avg_rating', 'team_avg_kill_death', 'team_avg_kill_round',
               'team_avg_rounds_with_kills', 'team_avg_kill_death_difference', 'team_avg_total_opening_kills',
               'team_avg_total_opening_deaths', 'team_avg_opening_kill_ratio', 'team_avg_opening_kill_rating',
               'team_avg_team_win_percent_after_first_kill', 'team_avg_first_kill_in_won_rounds']
    avg_df = pd.DataFrame(columns=columns)

    p1_df.columns = columns
    p2_df.columns = columns
    p3_df.columns = columns
    p4_df.columns = columns
    p5_df.columns = columns

    new = p1_df.add(p2_df, fill_value=0)
    new = new.add(p3_df, fill_value=0)
    new = new.add(p4_df, fill_value=0)
    new = new.add(p5_df, fill_value=0)

    avg_df = new.div(5)
    avg_df['map_id'] = players_df['map_id']

    team1_df = avg_df[:][::2]
    team2_df = avg_df[:][1::2]

    teams_df = pd.merge(team1_df, team2_df, on=['map_id'])
    train_df = pd.merge(train_df, teams_df, on=['map_id'])

    encode_maps = {
        "map_name": {"Ancient": 0, "Inferno": 1, "Nuke": 2, "Mirage": 3, "Overpass": 4, "Dust2": 5, "Vertigo": 6}}

    train_df = train_df.replace(encode_maps)

    test_df_new = pd.merge(test_df, teams_df, on=['map_id'])
    test_df_new = test_df_new.replace(encode_maps)

    return train_df, test_df_new
