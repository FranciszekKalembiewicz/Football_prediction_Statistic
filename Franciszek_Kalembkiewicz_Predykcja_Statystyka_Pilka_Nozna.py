import numpy as np
import pandas as pd
import random
from collections import defaultdict

# Funkcja generująca wyniki dla meczów
def final_score_probability_logistic(elo_club_1, elo_club_2, draw_factor=0.53, home_game=True):
    # Przewaga grania u siebie (dla mnie jest to 20%, lecz można to zmienić, by usprawnić program)
    if home_game:
        elo_club_1 += elo_club_1 / 5

    # Prawdopodobieństwo na wygraną drużyn
    # Prawdopodobieństwo remisu (zmienną draw_factor ustaliłem samemu, gdyż taka wartość najbardziej odzwierciedla rzeczywistość)
    p_win_1 = 1 / (1 + np.exp(-(elo_club_1 - elo_club_2) / 400))
    p_win_2 = 1 - p_win_1
    p_draw = draw_factor * (1 - abs(p_win_1 - p_win_2))

    # Poprawka na to by wszystkie prawdopodobieństwa sumowały się do 1
    total_prob = p_win_1 + p_win_2 + p_draw
    p_win_1 /= total_prob
    p_win_2 /= total_prob
    p_draw /= total_prob

    # Liczenie punktów oczekiwanych dla drużyn
    p_exp_points_1 = p_win_1 * 3 + p_draw * 1 + p_win_2 * 0
    p_exp_points_2 = p_win_2 * 3 + p_draw * 1 + p_win_1 * 0

    return p_win_1, p_draw, p_win_2, p_exp_points_1, p_exp_points_2


# Funkcja generująca mecze na podstawie drużyn
def generate_matches(df):
    clubs = df["Club"].values
    num_clubs = len(clubs)

    #Lista meczów z podziałem na rundy wiosna/jesień
    first_half = []
    second_half = []

    # Generowanie meczów (każdy z każdym raz u siebie ran na wyjeździe)
    for i in range(num_clubs):
        for j in range(i + 1, num_clubs):
            home = clubs[i]
            away = clubs[j]

            # Generujemy spotkania raz u siebie w pierwszej rundzie raz u przeciwnika w drugiej
            first_half.append((home, away, 'home'))
            second_half.append((away, home, 'away'))

    match_day_sezon = first_half + second_half
    return match_day_sezon


# Symulujemy mecze i losujemy wyniki na bazie prawdopodobieństw
def simulate_season(df):
    Premier_League_Sezon = generate_matches(df)
    expected_points = {club: 0 for club in df["Club"].values}
    actual_points = {club: 0 for club in df["Club"].values}

    # Symulacja meczów
    for match in Premier_League_Sezon:
        home_team = match[0]
        away_team = match[1]
        home_game = match[2] == 'home'

        # Pobieranie rankingów ELO dla drużyn
        elo_home = df[df["Club"] == home_team]["Elo"].values[0]
        elo_away = df[df["Club"] == away_team]["Elo"].values[0]

        # Obliczanie oczekiwanych punktów dla drużyn
        p_win_home, p_draw, p_win_away, p_exp_home, p_exp_away = final_score_probability_logistic(elo_home, elo_away, home_game=home_game)

        # Losowanie wyniku na podstawie prawdopodobieństw
        outcome = random.choices(
            ['home_win', 'draw', 'away_win'],
            weights=[p_win_home, p_draw, p_win_away],
            k=1
        )[0]

        # Przyznawanie punktów na podstawie wyniku meczu
        if outcome == 'home_win':
            actual_points[home_team] += 3
        elif outcome == 'away_win':
            actual_points[away_team] += 3
        else:
            actual_points[home_team] += 1
            actual_points[away_team] += 1

        # Dodanie oczekiwanych punktów
        expected_points[home_team] += p_exp_home
        expected_points[away_team] += p_exp_away

    # Tworzenie tabeli punktowej
    sorted_teams = sorted(actual_points.items(), key=lambda x: x[1], reverse=True)
    actual_table = pd.DataFrame(sorted_teams, columns=["Club", "Actual Points"])
    actual_table["Rank"] = range(1, len(actual_table) + 1)

    # Porównanie oczekiwanych punktów z rzeczywistymi
    expected_table = pd.DataFrame(list(expected_points.items()), columns=["Club", "Expected Points"])
    comparison = pd.merge(actual_table, expected_table, on="Club")
    comparison["Difference"] = comparison["Actual Points"] - comparison["Expected Points"]

    return comparison

# Tutaj jest program do predykcji spotkań "pucharowych" gdzie rozgrywany jest dwumecz
def simulate_two_legged_tie(df, team1, team2, simulations=10000, verbose=False):
    if team1 not in df["Club"].values or team2 not in df["Club"].values:
        raise ValueError("Jedna z drużyn nie istnieje w DataFrame.")

    team1_advances = 0
    team2_advances = 0
    penalties_used = 0

    for _ in range(simulations):
        total_goals_team1 = 0
        total_goals_team2 = 0

        # Mecz 1: team1 u siebie
        elo1 = df[df["Club"] == team1]["Elo"].values[0]
        elo2 = df[df["Club"] == team2]["Elo"].values[0]
        p1_home, p_draw, p2_away, *_ = final_score_probability_logistic(elo1, elo2, home_game=True)
        result_1 = random.choices(['home_win', 'draw', 'away_win'], weights=[p1_home, p_draw, p2_away])[0]

        if result_1 == 'home_win':
            total_goals_team1 += 2
        elif result_1 == 'away_win':
            total_goals_team2 += 2
        else:
            total_goals_team1 += 1
            total_goals_team2 += 1

        # Mecz 2: team2 u siebie
        p2_home, p_draw, p1_away, *_ = final_score_probability_logistic(elo2, elo1, home_game=True)
        result_2 = random.choices(['home_win', 'draw', 'away_win'], weights=[p2_home, p_draw, p1_away])[0]

        if result_2 == 'home_win':
            total_goals_team2 += 2
        elif result_2 == 'away_win':
            total_goals_team1 += 2
        else:
            total_goals_team1 += 1
            total_goals_team2 += 1

        if total_goals_team1 > total_goals_team2:
            team1_advances += 1
        elif total_goals_team1 < total_goals_team2:
            team2_advances += 1
        else:
            penalties_used += 1
            winner = random.choice([team1, team2])
            if winner == team1:
                team1_advances += 1
            else:
                team2_advances += 1

    # Tworzenie tabeli podsumowującej
    results_table = pd.DataFrame({
        "Zespół": [team1, team2],
        "Liczba awansów": [team1_advances, team2_advances],
        "Procent awansów": [f"{(team1_advances / simulations) * 100:.2f}%",
                            f"{(team2_advances / simulations) * 100:.2f}%"]
    })

    print(f"\nWyniki symulacji dwumeczu: {team1} vs {team2} (n = {simulations})")
    print(results_table.to_string(index=False))
    print(f"\nRzuty karne decydowały w: {penalties_used} z {simulations} symulacji "
          f"({penalties_used / simulations * 100:.2f}%)\n")

    return results_table


# Symulacja 100 sezonów (Powinno to spowodować przybliżenie wartości do expected points, lecz z ciekawości i tak to sprawdzę)
def simulate_multiple_seasons(df, num_simulations=100):
    clubs = df["Club"].values
    total_points = {club: 0 for club in clubs}
    total_rank = {club: 0 for club in clubs}

    for _ in range(num_simulations):
        # Symulacja sezonu
        comparison_table = simulate_season(df)

        # Sumowanie punktów i pozycji
        for _, row in comparison_table.iterrows():
            club = row["Club"]
            total_points[club] += row["Actual Points"]
            total_rank[club] += row["Rank"]

    # Obliczanie średnich
    avg_points = {club: total_points[club] / num_simulations for club in clubs}
    avg_rank = {club: total_rank[club] / num_simulations for club in clubs}

    # Tworzenie tabeli wyników
    results = []
    for club in clubs:
        results.append({
            "Club": club,
            "Średnie Punkty": avg_points[club],
            "Średnia Pozycja": avg_rank[club],
            # Usunięto odwołanie do "Expected Points" które nie istnieje w oryginalnym df
        })

    # Sortowanie po średnich punktach
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Średnie Punkty", ascending=False)
    results_df["Rank"] = range(1, len(results_df) + 1)  # Nowe pozycje rankingowe

    return results_df




# Wczytywanie danych
df = pd.read_csv("clubelo-club-rankings.csv")
df = df.iloc[:643]  # Przycinamy data frame, ponieważ od 643 linijki dane się powtarzają (zapętlają)

#Symulacja sezonu piłkarskiego
#Premier League
print("\n\n---------------------------Premier League---------------------------")
df_ENG = df[df["Country"] == "ENG"]  # Filtrowanie drużyn z Anglii
df_Premier_League = df_ENG.sort_values(by="Elo", ascending=False).head(20)
# Symulacja sezonu z losowaniem wyników
comparison_table = simulate_season(df_Premier_League)
print(comparison_table)

#Ekstraklasa
print("\n\n----------------------------Ekstraklasa----------------------------")
df_POL = df[df["Country"] == "POL"]  # Filtrowanie drużyn z Anglii
df_Ekstraklasa = df_POL.sort_values(by="Elo", ascending=False).head(18)
# Symulacja sezonu z losowaniem wyników
comparison_table = simulate_season(df_Ekstraklasa)
print(comparison_table)

#Symulacja spotkań pucharowych
#Liga konferencji
simulate_two_legged_tie(df, "Legia", "Chelsea")
simulate_two_legged_tie(df, "Jagiellonia", "Betis")



#Symulacja 100 sezonów piłkarkich
# Premier League
print("\n\n---------------------------Premier League (100 symulacji)---------------------------")
df_ENG = df[df["Country"] == "ENG"]
df_Premier_League = df_ENG.sort_values(by="Elo", ascending=False).head(20)
final_table = simulate_multiple_seasons(df_Premier_League, num_simulations=100)
print(final_table)

# Ekstraklasa
print("\n\n----------------------------Ekstraklasa (100 symulacji)----------------------------")
df_POL = df[df["Country"] == "POL"]
df_Ekstraklasa = df_POL.sort_values(by="Elo", ascending=False).head(18)
final_table = simulate_multiple_seasons(df_Ekstraklasa, num_simulations=100)
print(final_table)
