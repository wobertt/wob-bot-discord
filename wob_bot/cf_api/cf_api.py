"""
Commands for interacting with the Codeforces API.
"""

import requests
import pandas as pd


def make_request(api_endpoint: str):
    r = requests.get(f"https://codeforces.com/api/{api_endpoint}")
    r.raise_for_status()
    return r.json()["result"]


def get_contest_standings(
    contest_id: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Example:
    contest, problems, participants, problem_results = get_contest_standings(2033)

    CONTEST:
    {
        "id": 2033,
        "name": "Codeforces Round 981 (Div. 3)",
        "type": "ICPC",
        "phase": "FINISHED",
        "frozen": False,
        "durationSeconds": 8100,
        "startTimeSeconds": 1729780500,
        "relativeTimeSeconds": 48755654,
    }

    PROBLEMS:
       contestId label                                   name         type  rating                                               tags
    0       2033     A                    Sakurako and Kosuke  PROGRAMMING     800    [constructive algorithms, implementation, math]
    1       2033     B                     Sakurako and Water  PROGRAMMING     900     [brute force, constructive algorithms, greedy]
    2       2033     C                  Sakurako's Field Trip  PROGRAMMING    1400                         [dp, greedy, two pointers]
    3       2033     D                   Kousuke's Assignment  PROGRAMMING    1300           [data structures, dp, dsu, greedy, math]
    4       2033     E  Sakurako, Kosuke, and the Permutation  PROGRAMMING    1400  [brute force, data structures, dfs and similar...
    5       2033     F                         Kosuke's Sloth  PROGRAMMING    1800                 [brute force, math, number theory]
    6       2033     G                    Sakurako and Chefir  PROGRAMMING    2200  [data structures, dfs and similar, dp, greedy,...

    PARTICIPANTS:
                     handle   rank  points  penalty  successfulHackCount  unsuccessfulHackCount
    0                  Valj      1     7.0      287                    0                      0
    1              MR.Shiva      2     7.0      300                    0                      0
    2           Salty_jelly      3     7.0      363                    0                      0
    3                  Andy      4     7.0      690                    0                      0
    4            beunique18      5     6.0      111                    0                      0
    ...                 ...    ...     ...      ...                  ...                    ...
    13933            lamoon  13780     0.0        0                    0                      0
    13934            vinmen  13780     0.0        0                    0                      0
    13935  sanjeevkumarray1  13780     0.0        0                    0                      0
    13936          Abdo_S_S  13780     0.0        0                    0                      0
    13937       beingbetter  13780     0.0        0                    0                      0
    [13938 rows x 6 columns]

    PROBLEM_RESULTS:
           problem_index       handle  points  rejectedAttemptCount   type  bestSubmissionTimeSeconds
    0                  0         Valj     1.0                     0  FINAL                      186.0
    1                  1         Valj     1.0                     0  FINAL                      289.0
    2                  2         Valj     1.0                     1  FINAL                     1346.0
    3                  3         Valj     1.0                     0  FINAL                     1608.0
    4                  4         Valj     1.0                     0  FINAL                     1827.0
    ...              ...          ...     ...                   ...    ...                        ...
    97561              2  beingbetter     0.0                     4  FINAL                        NaN
    97562              3  beingbetter     0.0                     2  FINAL                        NaN
    97563              4  beingbetter     0.0                     0  FINAL                        NaN
    97564              5  beingbetter     0.0                     0  FINAL                        NaN
    97565              6  beingbetter     0.0                     0  FINAL                        NaN
    [97566 rows x 5 columns]
    """

    def parse_standings_row(row: dict) -> tuple[dict, pd.DataFrame]:
        """Parse one row of the 'rows' section in /api/contest.standings."""
        # only support single-party teams for now
        assert len(row["party"]["members"]) == 1
        handle = row["party"]["members"][0]["handle"]

        participant_info = {"handle": handle} | {
            key: row[key]
            for key in (
                "rank",
                "points",
                "penalty",
                "successfulHackCount",
                "unsuccessfulHackCount",
            )
        }
        problem_results = pd.DataFrame(
            {"handle": handle} | res for res in row["problemResults"]
        )
        return participant_info, problem_results

    json = make_request(f"contest.standings?contestId={contest_id}")

    contest = json["contest"]
    problems = pd.DataFrame(json["problems"]).rename(columns={"index": "label"})

    standings = tuple(parse_standings_row(row) for row in json["rows"])
    participants = pd.DataFrame(p_info for p_info, _ in standings)
    problem_results = pd.concat(results for _, results in standings)

    # Some fixups.
    problem_results = problem_results.reset_index(names="problem_index")

    return contest, problems, participants, problem_results


def get_rating_changes(contest_id: int) -> pd.DataFrame:
    """
    Example:
    print(get_rating_changes(2033))

                    handle   rank  ratingUpdateTimeSeconds  oldRating  newRating
    0             ponjuice      1               1729788600       1422       1907
    1           AIinLoh420      2               1729788600          0        877
    2              tktkqw9      3               1729788600        754       1407
    3         baoziiii1224      4               1729788600       1246       1727
    4        god_bless_you      5               1729788600        841       1424
    ...                ...    ...                      ...        ...        ...
    22427         rohith_5  21718               1729788600        579        705
    22428  Meghamala_Pilli  21718               1729788600        590        715
    22429         Abdo_S_S  21718               1729788600       1084        964
    22430   KLU_2200030987  21718               1729788600        358        574
    22431           Mr_mit  21718               1729788600          0        351

    [22432 rows x 5 columns]
    """
    json = make_request(f"contest.ratingChanges?contestId={contest_id}")
    rating_changes = pd.DataFrame(json).drop(columns=["contestId", "contestName"])
    return rating_changes


# Functions to confirm that the output looks right.
def test_contest_standings():
    contest, problems, participants, problem_results = get_contest_standings(2033)
    print(contest)
    print(problems)
    print(participants)
    print(problem_results)


def test_rating_changes():
    rating_changes = get_rating_changes(2033)
    print(rating_changes)


if __name__ == "__main__":
    # test_contest_standings()
    test_rating_changes()
