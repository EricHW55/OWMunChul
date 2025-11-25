import re
import os
import pandas as pd
import numpy as np


class OWFeatureBuilder:
    """
    - ow_stats.csv 를 읽어서
      * hero 이름 정규화 (숫자 suffix 제거, 모르면 unknown)
      * hero 원핫 인코딩
      * 매치/팀 단위 비율 피쳐
      * K/D, 데미지/킬, 힐/데스 같은 파생 피쳐
    - 결과를 새 CSV로 저장
    """

    # 원본 영웅 목록(숫자 붙은 버전 포함)
    RAW_HERO_NAMES = [
        'ana', 'ashe', 'ashe2', 'baptiste', 'bastion', 'bastion2', 'bastion3',
        'brigitte', 'brigitte2', 'cassidy', 'cassidy2', 'cassidy3',
        'doomfist', 'doomfist2', 'dva', 'dva2', 'echo', 'freja', 'genji',
        'hanzo', 'hazard', 'illari', 'illari2', 'junker_queen', 'junkrat',
        'juno', 'kiriko', 'kiriko2', 'lifeweaver', 'lucio', 'lucio2',
        'mauga', 'mei', 'mercy', 'moira', 'orisa', 'orisa2', 'pharah',
        'pharah2', 'ramattra', 'reaper', 'reinhardt', 'roadhog', 'sigma',
        'sojourn', 'soldier', 'sombra', 'sombra2', 'symmetra', 'symmetra2',
        'torbjorn', 'tracer', 'venture', 'widowmaker', 'widowmaker2',
        'winston', 'wrecking_ball', 'wuyang', 'zarya', 'zenyatta', 'zenyatta2'
    ]

    # 숫자 suffix 제거한 base 이름들
    BASE_HERO_NAMES = sorted(
        set(re.sub(r"\d+$", "", h) for h in RAW_HERO_NAMES)
    )

    def __init__(self, input_csv="ow_stats.csv",
                 output_csv="ow_stats_features.csv"):
        self.input_csv = input_csv
        self.output_csv = output_csv

        # unknown까지 포함한 최종 hero 카테고리
        self.hero_categories = self.BASE_HERO_NAMES + ["unknown"]

    # ---------- 내부 유틸 ----------

    def _normalize_hero_name(self, name: str) -> str:
        """
        'ashe2' -> 'ashe', 'lucio3' -> 'lucio' 처럼 숫자 suffix 제거.
        리스트에 없는 이름이면 'unknown' 으로.
        """
        if not isinstance(name, str) or name == "":
            return "unknown"

        base = re.sub(r"\d+$", "", name)
        if base in self.BASE_HERO_NAMES:
            return base
        if name == "unknown":
            return "unknown"
        # 혹시 이상한 이름이 들어오면 걍 unknown 처리
        return "unknown"

    def _ensure_numeric(self, df: pd.DataFrame, cols):
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        return df

    # ---------- 메인 처리 ----------

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.input_csv)
        return df

    def add_hero_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        # 영웅 이름 정규화
        df["hero_norm"] = df["hero"].astype(str).apply(self._normalize_hero_name)

        # 원핫 인코딩 (hero_<name> 열 생성)
        for h in self.hero_categories:
            col = f"hero_{h}"
            df[col] = (df["hero_norm"] == h).astype(int)

        return df

    def add_match_and_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        src_image 단위(한 경기 스샷) / (src_image, team) 단위로
        합계와 비율 피쳐 생성
        """
        # 숫자 컬럼 정리
        df = self._ensure_numeric(
            df,
            ["kills", "assists", "deaths", "damage", "heal", "mitig"]
        )

        # -------- 1) 매치 전체 합계 (src_image 기준) --------
        group_match = df.groupby("src_image")

        df["match_total_kills"] = group_match["kills"].transform("sum")
        df["match_total_deaths"] = group_match["deaths"].transform("sum")
        df["match_total_damage"] = group_match["damage"].transform("sum")
        df["match_total_heal"] = group_match["heal"].transform("sum")

        # 0으로 나누기 방지
        df["match_total_kills_clip"] = df["match_total_kills"].clip(lower=1)
        df["match_total_deaths_clip"] = df["match_total_deaths"].clip(lower=1)
        df["match_total_damage_clip"] = df["match_total_damage"].clip(lower=1)
        df["match_total_heal_clip"] = df["match_total_heal"].clip(lower=1)

        # 매치 전체 대비 비율
        df["kill_share_match"] = df["kills"] / df["match_total_kills_clip"]
        df["death_share_match"] = df["deaths"] / df["match_total_deaths_clip"]
        df["damage_share_match"] = df["damage"] / df["match_total_damage_clip"]
        df["heal_share_match"] = df["heal"] / df["match_total_heal_clip"]

        # -------- 2) 팀 단위 합계 (src_image, team 기준) --------
        group_team = df.groupby(["src_image", "team"])

        df["team_total_kills"] = group_team["kills"].transform("sum")
        df["team_total_deaths"] = group_team["deaths"].transform("sum")
        df["team_total_damage"] = group_team["damage"].transform("sum")
        df["team_total_heal"] = group_team["heal"].transform("sum")

        df["team_total_kills_clip"] = df["team_total_kills"].clip(lower=1)
        df["team_total_deaths_clip"] = df["team_total_deaths"].clip(lower=1)
        df["team_total_damage_clip"] = df["team_total_damage"].clip(lower=1)
        df["team_total_heal_clip"] = df["team_total_heal"].clip(lower=1)

        # 팀 대비 비율
        df["kill_share_team"] = df["kills"] / df["team_total_kills_clip"]
        df["death_share_team"] = df["deaths"] / df["team_total_deaths_clip"]
        df["damage_share_team"] = df["damage"] / df["team_total_damage_clip"]
        df["heal_share_team"] = df["heal"] / df["team_total_heal_clip"]

        # -------- 3) 개인 파생 피쳐 (K/D, 킬당 딜, 데스당 힐 등) --------
        deaths_clip = df["deaths"].clip(lower=1)
        kills_clip = df["kills"].clip(lower=1)

        df["kills_per_death"] = df["kills"] / deaths_clip          # 목숨당 킬
        df["damage_per_kill"] = df["damage"] / kills_clip          # 킬당 딜
        df["heal_per_death"] = df["heal"] / deaths_clip            # 데스당 힐
        df["kda"] = (df["kills"] + df["assists"]) / deaths_clip    # (킬+어시)/데스

        return df

    def build_features(self) -> pd.DataFrame:
        """
        전체 파이프라인 실행 후 DataFrame 반환
        """
        df = self.load()
        df = self.add_hero_onehot(df)
        df = self.add_match_and_team_features(df)

        # 중간 clip 컬럼들 안 쓰고 싶으면 여기서 삭제해도 됨
        drop_cols = [
            "match_total_kills_clip", "match_total_deaths_clip",
            "match_total_damage_clip", "match_total_heal_clip",
            "team_total_kills_clip", "team_total_deaths_clip",
            "team_total_damage_clip", "team_total_heal_clip",
        ]
        df = df.drop(columns=drop_cols)

        return df

    def save(self):
        """
        build_features() 호출 후 output_csv 로 저장
        """
        df = self.build_features()
        df.to_csv(self.output_csv, index=False)
        print(f"[DONE] Saved features to {self.output_csv}")


if __name__ == "__main__":
    builder = OWFeatureBuilder(
        input_csv="ow_stats.csv",
        output_csv="ow_stats_features.csv",
    )
    builder.save()