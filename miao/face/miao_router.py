import pandas as pd
from generate_query_features import generate_features
import lightgbm as lgb
import joblib

class MiaoRouter():

    def __init__(self):
        self.feature_cols = [
            'len_short','len_medium','len_long','is_yesno_be','is_can_could',
            'is_definition_what','is_count_measure','is_other','is_who_which',
            'is_time_when','is_compare','is_location_where','is_exists_avail',
            'is_how_other','is_price_cost','is_reason_why',
            'ans_yes_no','ans_boolean_choice','ans_quantity','ans_other',
            'ans_comparison','ans_entity_name','ans_procedure','ans_time_date',
            'ans_location','ans_reason_explain','ans_list_set','has_number',
            'time_date','time_month','time_year','is_vehicle','is_plant',
            'is_food','is_animal','ent_single','ent_multiple','ent_none',
        ]

        self.clf_loaded = joblib.load("lgbm_full.pkl")

    def route(self, queries):
        df = pd.DataFrame({"query": queries})
        out = generate_features(df)

        X_out = out[self.feature_cols]
        
        out['y_pred'] = self.clf_loaded.predict_proba(X_out)[:, 1]
        high_confidence_index = out[out["y_pred"] >= 0.61].index.to_list()

        return high_confidence_index


# mr = MiaoRouter()
# queries = df["query"].values
# mr.route(queries)
