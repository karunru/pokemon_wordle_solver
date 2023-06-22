from typing import List

import numpy as np
import pandas as pd
import streamlit as st


@st.cache
def make_base_df():
    df = pd.read_csv("./pokemon_names.csv")
    df = df[df["name"].str.len() == 5].reset_index(drop=True)
    char_freq_dict = (
        df["name"]
        .str.split("", expand=True)
        .iloc[:, 1:6]
        .unstack()
        .value_counts()
        .to_dict()
    )
    each_char_df = df["name"].str.split("", expand=True).iloc[:, 1:6]
    each_char_df.columns = [f"char_{i}" for i in range(5)]
    unique_char_idx = [
        np.unique(row, return_index=True)[1] for row in each_char_df.values
    ]
    for i in range(5):
        each_char_df[f"freq_{i}"] = each_char_df[f"char_{i}"].map(char_freq_dict)
    mask_arr = np.zeros(each_char_df[[f"freq_{i}" for i in range(5)]].shape, dtype=int)

    for idx, row in enumerate(mask_arr):
        row[unique_char_idx[idx]] = 1
    df["sum_char_freq"] = (
        mask_arr * each_char_df[[f"freq_{i}" for i in range(5)]].values
    ).sum(axis=1)
    df["rank"] = df["sum_char_freq"].rank(ascending=False)
    df = df.sort_values("rank").reset_index(drop=True)

    return df


def make_rank(names: pd.Series) -> pd.DataFrame:
    char_freq_dict = (
        names.str.split("", expand=True).iloc[:, 1:6].unstack().value_counts().to_dict()
    )

    each_char_df = names.str.split("", expand=True).iloc[:, 1:6]
    each_char_df.columns = [f"char_{i}" for i in range(5)]

    unique_char_idx = [
        np.unique(row, return_index=True)[1] for row in each_char_df.values
    ]

    for i in range(5):
        each_char_df[f"freq_{i}"] = each_char_df[f"char_{i}"].map(char_freq_dict)

    mask_arr = np.zeros(each_char_df[[f"freq_{i}" for i in range(5)]].shape, dtype=int)

    for idx, row in enumerate(mask_arr):
        row[unique_char_idx[idx]] = 1

    df = pd.DataFrame()
    df["name"] = names.copy()
    df["sum_char_freq"] = (
        mask_arr * each_char_df[[f"freq_{i}" for i in range(5)]].values
    ).sum(axis=1)
    df["rank"] = df["sum_char_freq"].rank(ascending=False)
    df = df.sort_values("rank").reset_index(drop=True)

    return df


def make_not_contain_chars_set(input_pokemon_name: str, wordle_result: str) -> set:
    input_char_list = [char for char in input_pokemon_name]
    output_not_contain_bool_list = [char == "0" for char in wordle_result]
    return set(np.array(input_char_list)[np.array(output_not_contain_bool_list)])


def make_contain_chars_dict(
    contain_chars_dict: str, input_pokemon_name: str, wordle_result: str
) -> list:
    input_char_list = [char for char in input_pokemon_name]
    output_contain_bool_list = [char in ["1", "2"] for char in wordle_result]
    keys = [i for i, x in enumerate(output_contain_bool_list) if x]
    values = np.array(input_char_list)[
        [i for i, x in enumerate(output_contain_bool_list) if x]
    ]

    for new_key, new_value in zip(keys, values):
        if new_key in contain_chars_dict.keys():
            contain_chars_dict[new_key] = contain_chars_dict[new_key] | set(new_value)
        else:
            contain_chars_dict[new_key] = set(new_value)

    return contain_chars_dict


def make_match_chars_dict(input_pokemon_name: str, wordle_result: str) -> list:
    input_char_list = [char for char in input_pokemon_name]
    output_contain_bool_list = [char == "2" for char in wordle_result]
    keys = [i for i, x in enumerate(output_contain_bool_list) if x]
    values = np.array(input_char_list)[
        [i for i, x in enumerate(output_contain_bool_list) if x]
    ].tolist()

    return dict(zip(keys, values))


def contain_and_not_here(
    df: pd.DataFrame, contain_chars_dict: dict, match_chars_dict: dict
):
    result = (df["name"] == "aaaa") | 1

    for key, value in contain_chars_dict.items():
        if len(_value := value - set(match_chars_dict.values())) > 0:
            result = result & (~df["name"].str[key].str.contains("|".join(_value)))
    return result


def contain_and_here(df: pd.DataFrame, match_chars_dict: dict):
    result = (df["name"] == "aaaa") | 1

    for key, value in match_chars_dict.items():
        result = result & (df["name"].str[key] == value)
    return result


def convert_set_list_to_set(l: List[set]) -> set:
    s = set()
    for _s in l:
        s = s | _s

    return s


def wordle_solver(
    input_pokemon_name: str,
    wordle_result: str,
    not_contain_chars_set: set,
    contain_chars_dict: dict,
    match_chars_dict: dict,
    _df: pd.DataFrame,
) -> pd.DataFrame:
    if input_pokemon_name == "aaaaa":
        return _df

    contain_chars_dict = make_contain_chars_dict(
        contain_chars_dict, input_pokemon_name, wordle_result
    )

    match_chars_dict = match_chars_dict | make_match_chars_dict(
        input_pokemon_name, wordle_result
    )

    not_contain_chars_set = not_contain_chars_set | make_not_contain_chars_set(
        input_pokemon_name, wordle_result
    )
    not_contain_chars_set = not_contain_chars_set - convert_set_list_to_set(
        contain_chars_dict.values()
    )
    not_contain_chars_set = not_contain_chars_set - set(match_chars_dict.values())

    contain_and_not_here_result = contain_and_not_here(
        _df, contain_chars_dict, match_chars_dict
    )

    contain_and_here_result = contain_and_here(_df, match_chars_dict)

    _df = _df[
        (~_df["name"].str.contains("|".join(not_contain_chars_set)))
        & _df["name"].str.contains(
            "^(?=.*"
            + ")(?=.*".join(convert_set_list_to_set(contain_chars_dict.values()))
            + ")"
        )
        & contain_and_not_here_result
        & contain_and_here_result
    ]
    _df = make_rank(_df["name"])

    return _df


def int_to_emoji_wordle_result(wordle_result: str) -> str:
    wordle_result_char_dict = {0: "⬛", 1: "🟨", 2: "🟩"}

    return "".join([wordle_result_char_dict[int(char)] for char in wordle_result])


def main():
    df = make_base_df()
    not_contain_chars_set = set([])
    contain_chars_dict = dict()
    match_chars_dict = dict()
    input_pokemon_name = "aaaaa"
    _df = df.copy()
    input_pokemon_name_history = []
    wordle_result_history = []

    st.header("pokemon wordle solver")
    input_pokemon_name = st.text_input(label="入力したポケモンの名前(5文字以外で終了)")
    if len(input_pokemon_name) != 5:
        st.warning("5文字のポケモンを入れてください")
    if input_pokemon_name not in df["name"].unique():
        st.warning("そのポケモンは存在しません")

    st.write(
        """
    wordleの結果
    
    0: 含まれていない(黒)
    
    1: 含まれている(黄)
    
    2: 含まれていて位置も一緒(緑)
    """
    )
    wordle_result = st.text_input(label="wordleの結果")
    if len(wordle_result) != 5:
        st.warning("5文字の結果を入れてください")

    if st.button("ENTER"):
        if "input_pokemon_name_history" not in st.session_state:
            st.session_state["input_pokemon_name_history"] = [input_pokemon_name]
        else:
            st.session_state["input_pokemon_name_history"].append(input_pokemon_name)

        if "wordle_result_history" not in st.session_state:
            st.session_state["wordle_result_history"] = [wordle_result]
        else:
            st.session_state["wordle_result_history"].append(wordle_result)

        for pokemon_name in st.session_state["input_pokemon_name_history"]:
            st.write(pokemon_name)

        for result in st.session_state["wordle_result_history"]:
            st.write(int_to_emoji_wordle_result(result))

        _df = wordle_solver(
            input_pokemon_name,
            wordle_result,
            not_contain_chars_set,
            contain_chars_dict,
            match_chars_dict,
            _df,
        )
        df_placeholder = st.empty()
        df_placeholder.table(_df)


if __name__ == "__main__":
    main()
