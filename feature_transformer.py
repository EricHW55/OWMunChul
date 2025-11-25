import re
import pandas as pd
import numpy as np


class OWFeatureTransformer:
    """
    - DataFrame(ow_stats 형식)을 받아서
      * hero 이름 정규화 (숫자 suffix 제거, 모르면 unknown)
      * hero 원핫 인코딩
      * 역할군(탱/딜/힐) 원핫 인코딩
      * 매치/팀 단위 합계 & 비율 피쳐
      * K/D, 데미지/킬, 데미지/데스, 힐/데스, KDA, 경감/데스,
        팀 킬관여율(kp_share_team), 경기 킬관여율(kp_share_match) 등 파생 피쳐
      * src_team, src_image, team, slot_index, hero, hero_norm 은 자동 drop
    - transform_file() 로 CSV -> CSV 변환도 가능
    """

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

    # -------- 역할군 매핑 (네가 직접 채우면 됨) --------
    # 기본 역할군: tank / damage / support
    # 필요하면 'flex', 'main_dps', 'off_dps', 'main_heal', 'flex_support' 같은 서브 역할도
    # ROLE_SUB_MAP 쪽에 네가 채워서 쓸 수 있게 틀만 만들어 둠.
    TANK_NAME = [
        'dva', 'doomfist', 'ramattra', 'wrecking_ball', 'roadhog', 'mauga', 
        'sigma', 'orisa', 'winston', 'zarya', 'junker_queen', 'hazard'
    ]
    DPS_NAME = [
        'genji', 'reaper', 'mei', 'bastion', 'venture', 'sojourn', 'soldier',
        'sombra', 'symmetra', 'ashe', 'echo', 'widowmaker', 'junkrat', 'cassidy', 
        'torbjorn', 'tracer', 'pharah', 'freja', 'hanzo'
    ]
    SUPPORT_NAME = [
        'lifeweaver', 'lucio', 'mercy', 'moira', 'baptiste', 'brigitte', 'ana',
        'wuyang', 'illari', 'zenyatta', 'juno', 'kiriko'
    ]
    
    ROLE_PRIMARY_MAP = {
        # Tank
        "dva": "tank",
        "doomfist": "tank",
        "ramattra": "tank",
        "wrecking_ball": "tank",
        "roadhog": "tank",
        "mauga": "tank",
        "sigma": "tank",
        "orisa": "tank",
        "winston": "tank",
        "zarya": "tank",
        "junker_queen": "tank",
        "hazard": "tank",

        # DPS
        "genji": "damage",
        "reaper": "damage",
        "mei": "damage",
        "bastion": "damage",
        "venture": "damage",
        "sojourn": "damage",
        "soldier": "damage",
        "sombra": "damage",
        "symmetra": "damage",
        "ashe": "damage",
        "echo": "damage",
        "widowmaker": "damage",
        "junkrat": "damage",
        "cassidy": "damage",
        "torbjorn": "damage",
        "tracer": "damage",
        "pharah": "damage",
        "freja": "damage",
        "hanzo": "damage",

        # Support
        "lifeweaver": "support",
        "lucio": "support",
        "mercy": "support",
        "moira": "support",
        "baptiste": "support",
        "brigitte": "support",
        "ana": "support",
        "wuyang": "support",
        "illari": "support",
        "zenyatta": "support",
        "juno": "support",
        "kiriko": "support",
    }
    
    

    def __init__(self):
        # unknown까지 포함한 최종 hero 카테고리
        self.hero_categories = self.BASE_HERO_NAMES + ["unknown"]

        # 모델 입력에서 항상 버릴 ID/메타 컬럼들
        self.id_cols_to_drop = [
            "src_team",
            "src_image",
            "team",
            "slot_index",
            "hero",
            "hero_norm",
        ]

    # ---------- 내부 유틸 ----------

    def _normalize_hero_name(self, name: str) -> str:
        """'ashe2' -> 'ashe', 'lucio3' -> 'lucio' 처럼 숫자 suffix 제거."""
        if not isinstance(name, str) or name == "":
            return "unknown"

        base = re.sub(r"\d+$", "", name)
        if base in self.BASE_HERO_NAMES:
            return base
        if name == "unknown":
            return "unknown"
        return "unknown"

    def _ensure_numeric(self, df: pd.DataFrame, cols):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        return df

    # ---------- 메인 처리 ----------

    def _add_hero_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        # 영웅 이름 정규화
        df["hero_norm"] = df["hero"].astype(str).apply(self._normalize_hero_name)

        # 원핫 인코딩 (hero_<name> 열 생성)
        for h in self.hero_categories:
            col = f"hero_{h}"
            df[col] = (df["hero_norm"] == h).astype(int)

        return df

    def _add_role_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        hero_norm 기준으로 역할군(tank/damage/support/unknown) 원핫 추가.
        ROLE_PRIMARY_MAP은 네가 직접 채워넣으면 됨.
        """

        def map_primary_role(hero_norm: str) -> str:
            h = str(hero_norm)
            # 혹시 suffix 안 지워진 값이 와도 한번 더 방어
            h = re.sub(r"\d+$", "", h)
            return self.ROLE_PRIMARY_MAP.get(h, "unknown")

        df["role_primary"] = df["hero_norm"].apply(map_primary_role)

        for r in ["tank", "damage", "support", "unknown"]:
            col = f"role_{r}"
            df[col] = (df["role_primary"] == r).astype(int)
            
        return df

    def _add_match_and_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        src_image 단위(한 경기 스샷) / (src_image, team) 단위로
        합계와 비율 & 파생 피쳐 생성
        """
        # 숫자 컬럼 정리
        df = self._ensure_numeric(
            df,
            ["kills", "assists", "deaths", "damage", "heal", "mitig"]
        )

        # -------- 1) 매치 전체 합계 (src_image 기준) --------
        if "src_image" not in df.columns:
            raise ValueError("DataFrame에 'src_image' 컬럼이 필요합니다.")

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

        # Kill Participation (경기 기준 킬 관여율)
        df["kp_share_match"] = (df["kills"] + df["assists"]) / df["match_total_kills_clip"]

        # -------- 2) 팀 단위 합계 (src_image, team 기준) --------
        if "team" not in df.columns:
            raise ValueError("DataFrame에 'team' 컬럼이 필요합니다.")

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

        # Kill Participation (팀 기준 킬 관여율)
        df["kp_share_team"] = (df["kills"] + df["assists"]) / df["team_total_kills_clip"]

        # -------- 3) 개인 파생 피쳐 (K/D, 킬당 딜, 데스당 힐 등) --------
        deaths_clip = df["deaths"].clip(lower=1)
        kills_clip = df["kills"].clip(lower=1)

        df["kills_per_death"] = df["kills"] / deaths_clip          # 목숨당 킬
        df["damage_per_kill"] = df["damage"] / kills_clip          # 킬당 딜
        df["damage_per_death"] = df["damage"] / deaths_clip        # 목숨당 딜량
        df["heal_per_death"] = df["heal"] / deaths_clip            # 데스당 힐
        df["mitig_per_death"] = df["mitig"] / deaths_clip          # 데스당 경감량
        df["kda"] = (df["kills"] + df["assists"]) / deaths_clip    # (킬+어시)/데스

        # 중간 clip 컬럼 정리
        drop_cols = [
            "match_total_kills_clip", "match_total_deaths_clip",
            "match_total_damage_clip", "match_total_heal_clip",
            "team_total_kills_clip", "team_total_deaths_clip",
            "team_total_damage_clip", "team_total_heal_clip",
            "role_primary"
        ]
        df = df.drop(columns=drop_cols)

        return df

    # ---------- 외부에서 쓰는 메인 API ----------

    def transform(self, df: pd.DataFrame, drop_id_cols: bool = True) -> pd.DataFrame:
        """
        ow_stats 형식 DataFrame -> feature DataFrame

        - df: 최소한
          ['src_team','src_image','team','slot_index','hero',
           'kills','assists','deaths','damage','heal','mitig', ...] 포함
        - drop_id_cols=True 이면
          [src_team,src_image,team,slot_index,hero,hero_norm] 제거 후 반환
        - win 컬럼이 있으면 그대로 유지되므로,
          target으로 바로 쓸 수 있음.
        """
        df = df.copy()

        df = self._add_hero_onehot(df)       # hero_* 원핫
        df = self._add_role_onehot(df)       # role_* 원핫
        df = self._add_match_and_team_features(df)

        if drop_id_cols:
            drop_cols = [c for c in self.id_cols_to_drop if c in df.columns]
            df = df.drop(columns=drop_cols)

        return df

    def transform_file(self, input_csv: str, output_csv: str,
                       drop_id_cols: bool = True):
        """
        CSV -> CSV 변환용 헬퍼
        """
        df = pd.read_csv(input_csv)
        out_df = self.transform(df, drop_id_cols=drop_id_cols)
        out_df.to_csv(output_csv, index=False)
        print(f"[DONE] Saved features to {output_csv}")


if __name__ == "__main__":
    tf = OWFeatureTransformer()
    tf.transform_file("ow_stats.csv", "ow_stats_features.csv", drop_id_cols=True)
