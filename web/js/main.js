var obj; 

document.getElementById("run-projection").addEventListener("click", ()=>{

		if(document.getElementById("run_type").checked) {
			starting_cap = parseFloat(document.getElementById("starting_cap").value);
			hpi_assumption = parseFloat(document.getElementById("hpi_assumption").value);
			prop_reins_perc = parseFloat(document.getElementById("prop_reins_perc").value);
			int_rate_curve = document.getElementById("int_rate_curve").value;
			starting_ann_pols = parseInt(document.getElementById("starting_ann_pols").value);
			starting_erm_pols = parseInt(document.getElementById("starting_erm_pols").value);
			ann_nb = parseInt(document.getElementById("ann_nb").value);
			ann_margin = parseFloat(document.getElementById("ann_margin").value);
			steps_py = parseInt(document.getElementById("steps_py").value);
			erm_nb = parseInt(document.getElementById("erm_nb").value);
			num_sims = parseInt(document.getElementById("num_sims").value);
			incl_defs = document.getElementById("incl_defs").checked;
			proj_term = parseInt(document.getElementById("proj_term").value);
			
			eel.get_random_name(starting_cap, hpi_assumption, prop_reins_perc, int_rate_curve, starting_ann_pols, starting_erm_pols, ann_nb, ann_margin, steps_py, erm_nb, num_sims, incl_defs, proj_term, "My Run");
		} else {
		
			model_of_interest = document.getElementById("model_of_interest").value;
			variable_of_interest = document.getElementById("variable_of_interest").value;
			basis_of_interest = document.getElementById("basis_of_interest").value;
			age_of_interest = parseInt(document.getElementById("age_of_interest").value);
			ltv_of_interest = parseFloat(document.getElementById("ltv_of_interest").value);
			aer_of_interest = parseFloat(document.getElementById("aer_of_interest").value);
			hpi_of_interest = parseFloat(document.getElementById("hpi_of_interest").value);
			output_loan_val = document.getElementById("output_loan_val").checked;
			output_prop_val = document.getElementById("output_prop_val").checked;
			output_int_nneg_val = document.getElementById("output_int_nneg_val").checked;
			output_mkt_nneg_val = document.getElementById("output_mkt_nneg_val").checked;
			output_int_agg_val = document.getElementById("output_int_agg_val").checked;
			output_mkt_agg_val = document.getElementById("output_mkt_agg_val").checked;
			output_ifrs_val = document.getElementById("output_ifrs_val").checked;

			eel.get_random_number(model_of_interest, variable_of_interest, basis_of_interest, age_of_interest, ltv_of_interest, aer_of_interest, hpi_of_interest, output_loan_val, output_prop_val, output_int_nneg_val, output_mkt_nneg_val, output_int_agg_val, output_mkt_agg_val, output_ifrs_val);
		}		
	},
	false);



var i = 0;
var width = 0;
eel.expose(prompt_alerts);
function prompt_alerts(run_output) {
  document.getElementById("perc").innerHTML = run_output + "%";
  if (i == 0) {
    i = 1;
    var elem = document.getElementById("myBar");
    var id = setInterval(frame, 10);
    function frame() {
      if (width >= run_output) {
        clearInterval(id);
        i = 0;
      } else {
        width++;
        elem.style.width = width + "%";
      }
    }
  }
}

eel.expose(display_results);
function display_results(new_run_id) {
	document.getElementById("graph").src = 'outputs/' + new_run_id + '/graph.png';
	document.getElementById("delete_run_btn").style.display = "block";
	document.getElementById("run_id").value = new_run_id;
	document.getElementById("myModal").style.display = "none";
	document.getElementById("perc").innerHTML = "0%";
	document.getElementById("myBar").style.width = "0%";
	width = 0;
}


function get_output_dirs() {
	eel.get_output_dirs();	
}

eel.expose(display_output_dirs);
function display_output_dirs(dirs) {
	document.getElementById('past_runs').innerHTML = dirs;
}

function show(run_id) {
	document.getElementById("graph").src = 'outputs/' + run_id + '/graph.png';
	document.getElementById("delete_run_btn").style.display = "block";
	document.getElementById("run_id").value = run_id;
	document.getElementsByClassName("graph_li").style.backgroundColor = "#3498db";
	document.getElementById("graph_" + run_id).style.backgroundColor = '#3498db';
}

document.getElementById("delete_run_btn").addEventListener("click", ()=>{

		eel.delete_run(document.getElementById('run_id').value);
		get_output_dirs();
		document.getElementById("graph").src = "";
		document.getElementById("delete_run_btn").style.display = "none";
		
	},
	false);


if (document.addEventListener) {
  document.addEventListener('contextmenu', function(e) {
    alert("You've tried to open context menu"); //here you draw your own menu
    e.preventDefault();
  }, false);
} else {
  document.attachEvent('oncontextmenu', function() {
    alert("You've tried to open context menu");
    window.event.returnValue = false;
  });
}


