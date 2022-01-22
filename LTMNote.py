from ERM import ERM
from Mortality import Mortality
from InterestRates import InterestRate
from BusinessOperations import BusinessOperations as BizOps
from matplotlib import pyplot as plt
import pandas as pd


class LTMNote:
    def __init__(self, erm_policies, int_rate_curve, steps_py, lapse_rate=0.01):
        self.erm_policies = erm_policies
        self.int_rate_curve = int_rate_curve
        self.steps_py = steps_py
        self.lapse_rate = lapse_rate

    def note_cumulative_pcfs(self, stress_level):
        if stress_level == 'AAA':
            tmp_hpi_assumption = 0
            tmp_prop_shock = 0.35
            tmp_lapse_rate_dwn=[0.005] * 104
            tmp_lapse_rate_up = [0.15] * 5 + [self.lapse_rate] * 99
        if stress_level == 'AA':
            tmp_hpi_assumption = 0.015
            tmp_prop_shock = 0.3
            tmp_lapse_rate_dwn=[0.005] * 104
            tmp_lapse_rate_up = [0.1] * 5 + [self.lapse_rate] * 99
        if stress_level == 'A':
            tmp_hpi_assumption = 0.02
            tmp_prop_shock = 0.25
            tmp_lapse_rate_dwn=[0.005] * 104
            tmp_lapse_rate_up = [0.08] * 5 + [self.lapse_rate] * 99
        if stress_level == 'BBB':
            tmp_hpi_assumption = 0.025
            tmp_prop_shock = 0.2
            tmp_lapse_rate_dwn=[0.005] * 104
            tmp_lapse_rate_up = [0.06] * 5 + [self.lapse_rate] * 99
        if stress_level == 'BB':
            tmp_hpi_assumption = 0.03
            tmp_prop_shock = 0.15
            tmp_lapse_rate_dwn=[0.005] * 104
            tmp_lapse_rate_up = [0.04] * 5 + [self.lapse_rate] * 99

        MdlEqRelDwn = ERM(self.int_rate_curve, self.steps_py, hpi_assumption=tmp_hpi_assumption, prop_shock=tmp_prop_shock, mortality_stress=f'{stress_level}_down', lapse_rate=tmp_lapse_rate_dwn)
        stress_dwn_cfs = MdlEqRelDwn.expected_redemption_cfs(policy.copy()).sum(axis=1)[:61]

        MdlEqRelUp = ERM(self.int_rate_curve, self.steps_py, hpi_assumption=tmp_hpi_assumption, prop_shock=tmp_prop_shock, mortality_stress=f'{stress_level}_up', lapse_rate=tmp_lapse_rate_up)
        stress_up_cfs = MdlEqRelUp.expected_redemption_cfs(policy.copy()).sum(axis=1)[:61]

        unshaped_note_cfs = pd.concat([stress_dwn_cfs, stress_up_cfs], axis=1).min(axis=1)

        MdlEqRelBE = ERM(self.int_rate_curve, self.steps_py, lapse_rate=[self.lapse_rate] * 104)
        be_cfs = MdlEqRelBE.expected_redemption_cfs(policy.copy()).sum(axis=1)[:61]

        pv_be_cfs = sum([cf * 1.04 ** (-t) for cf, t in zip(be_cfs, range(1, len(be_cfs) + 1))])
        print(f'pv_be_cfs = {pv_be_cfs}')
        pv_unshaped_note_cfs = sum([cf * 1.04 ** (-t) for cf, t in zip(unshaped_note_cfs, range(1, len(unshaped_note_cfs) + 1))])
        smoothed_note_cfs = be_cfs / pv_be_cfs * pv_unshaped_note_cfs

        return smoothed_note_cfs

    def note_pcfs(self, stress):
        ratings = {0: "AAA", 1: "AA", 2: "A", 3: "BBB", 4: "BB"}
        cum_pcfs = self.note_cumulative_pcfs(stress)
        if stress == 'AAA':
            return cum_pcfs
        else:
            prev_stress_index = list(ratings.keys())[list(ratings.values()).index(stress)] - 1
            prev_stress = ratings[prev_stress_index]
            prev_cum_pcfs = self.note_cumulative_pcfs(prev_stress)
            return cum_pcfs - prev_cum_pcfs

    def note_risk_free_value(self, note_id):
        ratings = {"Note 1": "AAA", "Note 2": "AA", "Note 3": "A", "Note 4": "BBB", "Note 5": "BB"}
        note_rating = ratings[note_id]
        note_pcfs = self.note_pcfs(note_rating)
        IntRateModel = InterestRate(self.int_rate_curve, 1, None)
        disc_curve = IntRateModel.discount_curve()
        return (note_pcfs * disc_curve.loc[note_pcfs.index, 'Rate']).sum()



if __name__ == '__main__':
    policy = BizOps.write_ltm_business(10000, steps_py=1)
    l = LTMNote(policy, 'flat_spot_4pc', 1)
    res = 0
    for i in range(1, 6):
        tmp = l.note_risk_free_value(f"Note {i}")
        res += tmp
        print(i, tmp)
    print(f'Total = {res}')

