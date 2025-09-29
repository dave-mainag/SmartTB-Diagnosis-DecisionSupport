import pickle
from flask import Flask, request, jsonify, render_template
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

app = Flask(__name__)

# Load pickled model
with open("tb_clinical_model.pkl", "rb") as f:
    model = pickle.load(f)

# Serve the main web page (index.html)
@app.route("/")
def index():
    # Renders the single-page app (index.html)
    return render_template("index.html")

# Endpoint to calculate disease probabilities based on user evidence and priors
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    evidence = data.get("evidence", {})
    priors = data.get("priors", {})

    # Only update CPDs for root nodes (exclude TB)
    root_nodes = ["HIV", "Pneumonia", "Bronchitis", "LungCancer"]
    for node in root_nodes:
        if node in priors:
            prob = priors[node]
            cpd = TabularCPD(variable=node, variable_card=2, values=[[1-prob],[prob]])
            model.add_cpds(cpd)

    # Recreate inference object
    infer = VariableElimination(model)

    # Convert boolean evidence to 0/1
    numeric_evidence = {k: int(v) for k, v in evidence.items()}

    diseases = ["TB", "Pneumonia", "Bronchitis", "LungCancer"]
    posterior = infer.query(variables=diseases, evidence=numeric_evidence)
    results = {disease: float(posterior[disease].values[1]) for disease in diseases}

    return jsonify(results)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=False)