import pandas as pd

from Annuity import Annuity
from Mortality import Mortality
from ERM import ERM
from flask import Flask, request, render_template
from io import StringIO


app = Flask(__name__)


@app.route("/")
def index():

	model_of_interest = request.args.get('model_of_interest', None)
	variable_of_interest = request.args.get('variable_of_interest', None)
	basis_of_interest = request.args.get('basis_of_interest', None)
	age_of_interest = int(request.args.get('age_of_interest', 0))
	ltv_of_interest = float(request.args.get('ltv_of_interest', 0))
	aer_of_interest = float(request.args.get('aer_of_interest', 0))
	hpi_of_interest = float(request.args.get('hpi_of_interest', 0))
	curve_of_interest = request.args.get('curve_of_interest', 0)
	prev_curves = request.args.get('prev_results', '')
	add_to_curr = request.args.get('add_to_curr', False)


	if not add_to_curr:
		prev_curves = ""
	if model_of_interest is not None:
		res_index, res_data, res_csv = curve_enquirer(model_of_interest, variable_of_interest, basis_of_interest, age_of_interest, ltv_of_interest, aer_of_interest, hpi_of_interest, curve_of_interest, prev_curves)
	else:
		res_index = ""
		res_data = ""
		res_csv = ""

	return render_template('index.html', res_index=res_index, res_data=res_data, prev_results=res_csv)

def curve_enquirer(model_of_interest, variable_of_interest, basis_of_interest, age_of_interest, ltv_of_interest, aer_of_interest, hpi_of_interest, curve_of_interest, prev_curves):
	print(f'curve_enquirer({model_of_interest}, {variable_of_interest}, {basis_of_interest}, {age_of_interest}, {ltv_of_interest}, {aer_of_interest}, {hpi_of_interest}, {curve_of_interest})')
	if model_of_interest == 'Annuity':

		# TODO: Add functionality to deal with deferreds

		df_ann_pols = pd.DataFrame([[1, age_of_interest, 1]], columns=['Policy ID', 'Age', 'Annual Amt'])
		AnnuityModel = Annuity('flat_spot_4pc', 1)
		res = pd.DataFrame()
		if variable_of_interest == 'Expected':
			if curve_of_interest == 'cfs':
				res[f'Expected CFs ({age_of_interest})'] = AnnuityModel.expected_cfs(df_ann_pols)
			else:
				res = "ERROR"
		else:
			res = "ERROR"

	if model_of_interest == 'LTM':

		ltv_of_interest /= 100
		aer_of_interest /= 100
		hpi_of_interest /= 100

		df_ltm_pols = pd.DataFrame([[age_of_interest, 1, 1/ltv_of_interest, aer_of_interest, 0]], columns=['Age', 'Loan Value', 'Property Value', 'AER', 'Spread'])
		EqRelModel = ERM('flat_spot_4pc')
		if variable_of_interest == 'Expected':
			res = pd.DataFrame()
			if curve_of_interest == 'red_cfs':
				res['Redemptions'] = EqRelModel.expected_redemption_cfs(df_ltm_pols)
			if curve_of_interest == 'int_nneg_val':
				# TODO: Remove prop shock hardcoding
				res['Intrinsic NNEG'] = EqRelModel.expected_int_nneg_cfs(df_ltm_pols, hpi_of_interest, None).iloc[:, 0]
			if curve_of_interest == 'red_net_of_int_nneg_cfs':
				# TODO: Remove prop shock hardcoding
				res['Redemptions Net of Int NNEG'] = EqRelModel.expected_redemption_cfs_net_of_int_nneg(df_ltm_pols, hpi_of_interest, None)
			if curve_of_interest == 'mkt_nneg_val':
				# TODO: Remove hpi vol and prop shock hardcoding
				res['Market NNEG'] = EqRelModel.expected_mkt_nneg_val(df_ltm_pols, hpi_of_interest, 0.12, None, True)
			if curve_of_interest == 'red_net_of_mkt_nneg_cfs':
				# TODO: Remove hpi vol and prop shock hardcoding
				res['Redemptions Net of Mkt NNEG'] = EqRelModel.expected_redemption_cfs_net_of_mkt_nneg(df_ltm_pols, hpi_of_interest, 0.12, None)
			if curve_of_interest == 'ifrs_val':
				exp_erm_red_cfs = EqRelModel.expected_redemption_cfs(df_ltm_pols)#, hpi_of_interest, 0.12)
				df_ltm_pols['Spread'] = EqRelModel.spread(df_ltm_pols, exp_erm_red_cfs).transpose()
				tmp_df_ltm_pols = df_ltm_pols.copy()
				MortalityModel = Mortality(1, None)
				tmp_prob_if = MortalityModel.prob_in_force()
				# TODO: Remove loop
				for t in range(0, 121 - age_of_interest):
					# TODO: Remove hpi vol and prop shock hardcoding
					res.loc[t, 'IFRS'] = EqRelModel.value(tmp_df_ltm_pols, hpi_of_interest, 0.12, None, t) * tmp_prob_if.loc[age_of_interest,  t]
					tmp_df_ltm_pols['Loan Value'] *= (1 + tmp_df_ltm_pols['AER'])
					tmp_df_ltm_pols['Age'] += 1
					tmp_df_ltm_pols['Property Value'] *= (1 + hpi_of_interest)
		elif variable_of_interest == "In Force":
			res = pd.DataFrame()
			if curve_of_interest == 'loan_val':
				res['Loan Value'] = [(1 + aer_of_interest) ** t for t in range(0, 51)]
			if curve_of_interest == 'prop_val':
				res['Property Value'] = [(1/ltv_of_interest) * (1 + hpi_of_interest) ** t for t in range(0, 51)]
			if curve_of_interest == 'int_nneg_val':
				res['Intrinsic NNEG'] = [max((1 + aer_of_interest) ** t - (1/ltv_of_interest) * (1 + hpi_of_interest) ** t, 0) for t in range(0, 51)]
			if curve_of_interest == 'mkt_nneg_val':
				# TODO: Remove hpi vol and prop shock hardcoding
				res['Market NNEG'] = EqRelModel.projected_nneg_mkt_value(df_ltm_pols, hpi_of_interest, 0.12, None).iloc[:51, 0]
			if curve_of_interest == 'loan_net_int_nneg_val':
				res['Loan Net of Int NNEG'] = [min((1 + aer_of_interest) ** t, (1/ltv_of_interest) * (1 + hpi_of_interest) ** t) for t in range(0, 51)]
			if curve_of_interest == 'loan_net_mkt_nneg_val':
				# TODO: Remove hpi vol and prop shock hardcoding
				res['Loan Net of Mkt NNEG'] = [(1 + aer_of_interest) ** t for t in range(0, 51)] - EqRelModel.projected_nneg_mkt_value(df_ltm_pols, hpi_of_interest, 0.12, None).iloc[:51, 0]
			if curve_of_interest == 'ifrs_val':
				exp_erm_red_cfs = EqRelModel.expected_redemption_cfs(df_ltm_pols)#, hpi_of_interest, 0.12)
				df_ltm_pols['Spread'] = EqRelModel.spread(df_ltm_pols, exp_erm_red_cfs).transpose()
				tmp_df_ltm_pols = df_ltm_pols.copy()
				# TODO: Remove loop
				for t in range(0, 121 - age_of_interest):
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
			res.columns=['qx Curve']
		elif variable_of_interest == 'IF Prob':
			res = pd.DataFrame(MortalityModel.prob_in_force().loc[age_of_interest])
			res.columns=['Prob IF']
		elif variable_of_interest == 'Prob Death in Year':
			res = pd.DataFrame(MortalityModel.prob_death().loc[age_of_interest])
			res.columns=['Prob Death']

	res = res.loc[:min(50, 120 - age_of_interest)]

	if prev_curves != "":
		pd_prev_curves = pd.read_csv(StringIO(prev_curves), index_col=0)
		if pd_prev_curves.shape[0] < res.shape[0]:
			merge_type = 'right'
		else:
			merge_type = 'left'
		res = res.merge(pd_prev_curves, left_index=True, right_index=True, how=merge_type)

	res_data = ""
	color_index = 0
	color_list = ['#03A9F4', '#9b59b6', '#2ecc71', '#e67e22', '#e74c3c', '#16a085', '#34495e']
	for col in res:
		res_data += "{label: '" + col + "', backgroundColor: '" + color_list[color_index] + "', borderColor: '" + color_list[color_index] + "', data: " + str(res[col].to_list()) + ", pointRadius: 0, },"
		color_index += 1



	return res.index.to_list(), res_data, res.to_csv()



if __name__ == '__main__':
	try:
		import googleclouddebugger

		googleclouddebugger.enable(
			breakpoint_enable_canary=True
		)
	except ImportError:
		pass
	app.run(host="127.0.0.1", port=8080, debug=True)
	#curve_enquirer('LTM', 'Expected', 'Best Estimate', 65, 40.0, 7.0, 3.0, 'red_cfs')
