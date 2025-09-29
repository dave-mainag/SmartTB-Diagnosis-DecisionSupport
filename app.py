import pickle
from flask import Flask, request, jsonify, render_template
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

app = Flask(__name__)

# Load pickled model
with open("tb_clinical_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    # Renders the single-page app (index.html)
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    evidence = data.get("evidence", {})  # expects dict like {"Cough": true, "HIV": true}
    priors = data.get("priors", {})      # optional sliders for Pneumonia/Bronchitis/LungCancer or HIV prior

    # If priors supplied, update those root CPDs (HIV, Pneumonia, Bronchitis, LungCancer)
    # Note: TB prior is now conditional on HIV, so do NOT set TB as unconditional prior.
    for node, prob in priors.items():
        # Only allow updating root nodes that actually are root in model
        cpd = TabularCPD(variable=node, variable_card=2, values=[[1-prob],[prob]])
        model.add_cpds(cpd)

    # Recreate infer after any CPD changes
    infer = VariableElimination(model)

    # Convert boolean evidence to 0/1 ints for pgmpy
    numeric_evidence = {k: int(v) for k, v in evidence.items()}

    diseases = ["TB", "Pneumonia", "Bronchitis", "LungCancer"]
    posterior = infer.query(variables=diseases, evidence=numeric_evidence)
    results = {disease: float(posterior[disease].values[1]) for disease in diseases}

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=False)