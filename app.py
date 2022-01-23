import eel
import random
from datetime import datetime

import pandas as pd

from InsuranceCompany import InsuranceCompany as IC
from matplotlib import pyplot as plt
from Annuity import Annuity
from Mortality import Mortality
from BusinessOperations import BusinessOperations as BizOps
from ERM import ERM
import uuid
import os
import shutil


eel.init('web')

@eel.expose
def get_random_name(starting_cap, hpi_assumption, prop_reins_perc, int_rate_curve, starting_ann_pols, starting_erm_pols, ann_nb, ann_margin, steps_py, erm_nb, num_sims, incl_defs, proj_term, run_name):
	run_id = call_run_sim(starting_cap, hpi_assumption, prop_reins_perc, int_rate_curve, starting_ann_pols, starting_erm_pols, ann_nb, ann_margin, steps_py, erm_nb, num_sims, incl_defs, proj_term, run_name)
	print(run_id)
	eel.display_results(run_id)

@eel.expose
def get_random_number(model_of_interest, variable_of_interest, basis_of_interest, age_of_interest, ltv_of_interest, aer_of_interest, hpi_of_interest, output_loan_val, output_prop_val, output_int_nneg_val, output_mkt_nneg_val, output_int_agg_val, output_mkt_agg_val, output_ifrs_val):
	print(f'call_curve_enquirer({model_of_interest}, {variable_of_interest}, {basis_of_interest}, {age_of_interest}, {ltv_of_interest}, {aer_of_interest}, {hpi_of_interest}, {output_loan_val}, {output_prop_val}, {output_int_nneg_val}, {output_mkt_nneg_val}, {output_int_agg_val}, {output_mkt_agg_val}, {output_ifrs_val})')
	run_id = call_curve_enquirer(model_of_interest, variable_of_interest, basis_of_interest, age_of_interest, ltv_of_interest, aer_of_interest, hpi_of_interest, output_loan_val, output_prop_val, output_int_nneg_val, output_mkt_nneg_val, output_int_agg_val, output_mkt_agg_val, output_ifrs_val)
	print(run_id)
	eel.display_results(run_id)

@eel.expose
def get_date():
	eel.prompt_alerts(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

@eel.expose
def run_status(status):
	eel.prompt_alerts(status)

@eel.expose
def get_output_dirs():
	res = "<ul id='past_run_list'>"
	output_dir = os.listdir('web/outputs')
	for item in output_dir:
		#if os.path.isdir(item):
		res += f"<li id='graph_{item}' class='graph_li'><img src='outputs/{item}/graph.png' class='past_run_graph' onclick='show(\"{item}\")'></li>"
	res += "</ul>"
	print(res)
	eel.display_output_dirs(res)



def call_run_sim(starting_cap, hpi_assumption, prop_reins_perc, int_rate_curve, starting_ann_pols, starting_erm_pols, ann_nb, ann_margin, steps_py, erm_nb, num_sims, incl_defs, proj_term, run_name):
	ann_portfolio = BizOps.write_annuity_business(starting_ann_pols, incl_defs, steps_py=steps_py)
	erm_portfolio = BizOps.write_ltm_business(starting_erm_pols, steps_py=steps_py)
	for i in range(1, num_sims + 1):
		eel.prompt_alerts(i / num_sims * 100)
		print(f'Running sim {i}')
		ModelIC = IC(starting_cap, int_rate_curve, hpi_assumption, ann_portfolio, ann_margin, erm_portfolio, ann_nb, erm_nb, incl_defs, prop_reins_perc, steps_py)
		if i == 1:
			df_projection = ModelIC.run_single_projection(proj_term * steps_py) / num_sims
		else:
			df_projection += ModelIC.run_single_projection(proj_term * steps_py) / num_sims

	for col in df_projection:
		plt.plot(df_projection.index, df_projection[col] / 1000000, label=col)

	run_id = uuid.uuid4()
	os.mkdir(f'web/outputs/{run_id}')

	df_projection.to_csv(f'web/outputs/{run_id}/balance_sheet.csv')
	if starting_ann_pols > 0:
		ann_portfolio.to_csv(f'web/outputs/{run_id}/initial_annuity_portfolio.csv')
	if starting_erm_pols > 0:
		erm_portfolio.to_csv(f'web/outputs/{run_id}/initial_erm_portfolio.csv')

	plt.xlabel('Time (Years)')
	plt.ylabel('£s (Million)')
	plt.legend()
	plt.title(run_name)
	plt.savefig(f'web/outputs/{run_id}/graph.png')
	plt.figure().clear()
	plt.close()
	plt.cla()
	plt.clf()

	get_output_dirs()

	return str(run_id)


def call_curve_enquirer(model_of_interest, variable_of_interest, basis_of_interest, age_of_interest, ltv_of_interest, aer_of_interest, hpi_of_interest, output_loan_val, output_prop_val, output_int_nneg_val, output_mkt_nneg_val, output_int_agg_val, output_mkt_agg_val, output_ifrs_val):

	if model_of_interest == 'Annuity':

		# TODO: Add functionality to deal with deferreds

		df_ann_pols = pd.DataFrame([[1, age_of_interest, 1]], columns=['Policy ID', 'Age', 'Annual Amt'])
		AnnuityModel = Annuity('flat_spot_4pc', 1)
		if variable_of_interest == 'Expected Cashflows':
			res = AnnuityModel.expected_cfs(df_ann_pols)
		else:
			res = "ERROR"

	if model_of_interest == 'LTM':

		ltv_of_interest /= 100
		aer_of_interest /= 100
		hpi_of_interest /= 100

		df_ltm_pols = pd.DataFrame([[age_of_interest, 1, 1/ltv_of_interest, aer_of_interest, 0]], columns=['Age', 'Loan Value', 'Property Value', 'AER', 'Spread'])
		EqRelModel = ERM('flat_spot_4pc')
		if variable_of_interest == 'Expected Cashflows':
			res = pd.DataFrame()
			if output_loan_val:
				res['Redemptions'] = EqRelModel.expected_redemption_cfs(df_ltm_pols)
			if output_int_nneg_val:
				# TODO: Remove prop shock hardcoding
				res['Intrinsic NNEG'] = EqRelModel.expected_int_nneg_cfs(df_ltm_pols, hpi_of_interest, None).iloc[:, 0]
			if output_int_agg_val:
				# TODO: Remove prop shock hardcoding
				res['Redemptions Net of Int NNEG'] = EqRelModel.expected_redemption_cfs_net_of_int_nneg(df_ltm_pols, hpi_of_interest, None)
			if output_mkt_nneg_val:
				# TODO: Remove hpi vol and prop shock hardcoding
				res['Market NNEG'] = EqRelModel.expected_mkt_nneg_cfs(df_ltm_pols, hpi_of_interest, 0.12, None, True)
			if output_mkt_agg_val:
				# TODO: Remove hpi vol and prop shock hardcoding
				res['Redemptions Net of Mkt NNEG'] = EqRelModel.expected_redemption_cfs_net_of_mkt_nneg(df_ltm_pols, hpi_of_interest, 0.12, None)
		elif variable_of_interest == "Projected Values":
			res = pd.DataFrame()
			if output_loan_val:
				res['Loan Value'] = [(1 + aer_of_interest) ** t for t in range(0, 51)]
			if output_prop_val:
				res['Property Value'] = [(1/ltv_of_interest) * (1 + hpi_of_interest) ** t for t in range(0, 51)]
			if output_int_nneg_val:
				res['Intrinsic NNEG'] = [max((1 + aer_of_interest) ** t - (1/ltv_of_interest) * (1 + hpi_of_interest) ** t, 0) for t in range(0, 51)]
			if output_mkt_nneg_val:
				# TODO: Remove hpi vol and prop shock hardcoding
				res['Market NNEG'] = EqRelModel.projected_nneg_mkt_value(df_ltm_pols, hpi_of_interest, 0.12, None).iloc[:51, 0]
			if output_int_agg_val:
				res['Loan Net of Int NNEG'] = [min((1 + aer_of_interest) ** t, (1/ltv_of_interest) * (1 + hpi_of_interest) ** t) for t in range(0, 51)]
			if output_mkt_agg_val:
				# TODO: Remove hpi vol and prop shock hardcoding
				res['Loan Net of Mkt NNEG'] = [(1 + aer_of_interest) ** t for t in range(0, 51)] - EqRelModel.projected_nneg_mkt_value(df_ltm_pols, hpi_of_interest, 0.12, None).iloc[:51, 0]
			if output_ifrs_val:
				exp_erm_red_cfs = EqRelModel.expected_redemption_cfs(df_ltm_pols)#, hpi_of_interest, 0.12)
				df_ltm_pols['Spread'] = EqRelModel.spread(df_ltm_pols, exp_erm_red_cfs).transpose()
				tmp_df_ltm_pols = df_ltm_pols.copy()
				# TODO: Remove loop
				for t in range(0, 51):
					# TODO: Remove hpi vol and prop shock hardcoding
					res.loc[t, 'IFRS'] = EqRelModel.value(tmp_df_ltm_pols, hpi_of_interest, 0.12, None, t)
					tmp_df_ltm_pols['Loan Value'] *= (1 + tmp_df_ltm_pols['AER'])
					tmp_df_ltm_pols['Age'] += 1
					tmp_df_ltm_pols['Property Value'] *= (1 + hpi_of_interest)

		else:
			res = "ERROR"

	if model_of_interest == 'Mortality':
		MortalityModel = Mortality(1, None)
		if variable_of_interest == 'qx Curve':
			res = pd.DataFrame(MortalityModel.q([x for x in range(age_of_interest, 121)]))
		elif variable_of_interest == 'IF Prob':
			res = pd.DataFrame(MortalityModel.prob_in_force().loc[age_of_interest])
		elif variable_of_interest == 'Prob Death in Year':
			res = pd.DataFrame(MortalityModel.prob_death().loc[age_of_interest])


	run_id = uuid.uuid4()
	os.mkdir(f'web/outputs/{run_id}')
	run_name = basis_of_interest + " " + model_of_interest + " " + variable_of_interest
	res = res.loc[:50]
	res.to_csv(f'web/outputs/{run_id}/curves.csv')

	for col in res:
		plt.plot(res.index, res[col], label=col)
	plt.xlabel('Time (Years)')
	plt.legend()
	plt.title(run_name)
	plt.savefig(f'web/outputs/{run_id}/graph.png')
	plt.figure().clear()
	plt.close()
	plt.cla()
	plt.clf()

	return str(run_id)

@eel.expose
def delete_run(run_id):
	shutil.rmtree(f'web/outputs/{run_id}')

if __name__ == '__main__':
	foo = call_curve_enquirer('LTM', 'Expected Cashflows', 'Best Estimate', 65, 40, 7, 3, True, True, True, True, True, True, True)
	print(foo)
	#print(get_output_dirs())
	eel.start('index.html')
	#get_output_dirs()