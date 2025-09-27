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
    evidence = data.get("evidence", {})
    priors = data.get("priors", {})

    # Update priors
    for disease, prob in priors.items():
        cpd = TabularCPD(variable=disease, variable_card=2, values=[[1 - prob], [prob]])
        model.add_cpds(cpd)

    # Convert evidence booleans (0/1)
    numeric_evidence = {k: int(v) for k, v in evidence.items()}

    results = {}
    for disease in ["TB", "Pneumonia", "Bronchitis", "LungCancer"]:
        q = VariableElimination(model).query(
            variables=[disease],
            evidence=numeric_evidence,
            show_progress=False
        )
        results[disease] = float(q.values[1])  # probability of "disease = 1"

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=False)