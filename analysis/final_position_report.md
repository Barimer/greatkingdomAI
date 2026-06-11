# Great Kingdom AI - Final Position Analysis Report

## 📊 Executive Summary

* **Analyzed Games**: 20 games
* **Average Territory**: BLUE 0.9 vs ORANGE 0.8
* **Average Liberties**: BLUE 1.8 vs ORANGE 1.4

## 🏆 Winning Tag Classification

| Tag Name | Game Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **DOMINANT_CAPTURE** | 15 | 75.0% | Opponent liberties <= 2 at termination |
| **NARROW_CAPTURE** | 0 | 0.0% | Opponent liberties 3~5 at termination |
| **COMEBACK_CAPTURE** | 5 | 25.0% | Winner territory < Loser territory at capture |
| **TERRITORY_PLUS_CAPTURE** | 0 | 0.0% | Winner territory >= Loser territory at capture |


## 📉 Difference Analysis

* **Average Territory Difference**: `BLUE +0.2`
* **Average Liberty Difference**: `BLUE +0.4`

## 🔍 Detailed Positions

| Game ID | Winner | Termination | Moves | BLUE Stones | ORANGE Stones | BLUE Libs | ORANGE Libs | BLUE Terr | ORANGE Terr | Tag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| game_001.json | BLUE | CAPTURE | 25 | 12 | 12 | 3 | 1 | 0 | 0 | `DOMINANT_CAPTURE` |
| game_002.json | BLUE | CAPTURE | 31 | 15 | 15 | 2 | 1 | 0 | 0 | `DOMINANT_CAPTURE` |
| game_003.json | ORANGE | CAPTURE | 14 | 7 | 6 | 1 | 1 | 0 | 0 | `DOMINANT_CAPTURE` |
| game_004.json | BLUE | CAPTURE | 21 | 10 | 10 | 2 | 1 | 0 | 0 | `DOMINANT_CAPTURE` |
| game_005.json | BLUE | CAPTURE | 19 | 9 | 9 | 2 | 1 | 0 | 0 | `DOMINANT_CAPTURE` |
| game_006.json | BLUE | CAPTURE | 61 | 30 | 30 | 3 | 1 | 2 | 6 | `COMEBACK_CAPTURE` |
| game_007.json | BLUE | CAPTURE | 25 | 12 | 12 | 2 | 1 | 0 | 0 | `DOMINANT_CAPTURE` |
| game_008.json | ORANGE | CAPTURE | 28 | 14 | 13 | 1 | 2 | 0 | 0 | `DOMINANT_CAPTURE` |
| game_009.json | ORANGE | CAPTURE | 24 | 12 | 11 | 1 | 3 | 0 | 0 | `DOMINANT_CAPTURE` |
| game_010.json | BLUE | CAPTURE | 27 | 13 | 13 | 1 | 1 | 4 | 0 | `DOMINANT_CAPTURE` |
| game_011.json | ORANGE | CAPTURE | 30 | 15 | 14 | 1 | 2 | 2 | 0 | `COMEBACK_CAPTURE` |
| game_012.json | BLUE | CAPTURE | 39 | 19 | 19 | 1 | 1 | 1 | 0 | `DOMINANT_CAPTURE` |
| game_013.json | BLUE | CAPTURE | 33 | 16 | 16 | 1 | 1 | 0 | 0 | `DOMINANT_CAPTURE` |
| game_014.json | BLUE | CAPTURE | 45 | 22 | 21 | 2 | 1 | 5 | 2 | `DOMINANT_CAPTURE` |
| game_015.json | BLUE | CAPTURE | 41 | 20 | 20 | 2 | 1 | 2 | 3 | `COMEBACK_CAPTURE` |
| game_016.json | ORANGE | CAPTURE | 50 | 25 | 24 | 1 | 2 | 0 | 2 | `DOMINANT_CAPTURE` |
| game_017.json | ORANGE | CAPTURE | 22 | 11 | 10 | 1 | 3 | 0 | 0 | `DOMINANT_CAPTURE` |
| game_018.json | BLUE | CAPTURE | 25 | 12 | 12 | 4 | 1 | 0 | 0 | `DOMINANT_CAPTURE` |
| game_019.json | ORANGE | CAPTURE | 52 | 26 | 25 | 1 | 1 | 2 | 0 | `COMEBACK_CAPTURE` |
| game_020.json | BLUE | CAPTURE | 39 | 19 | 19 | 3 | 1 | 0 | 2 | `COMEBACK_CAPTURE` |