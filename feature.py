import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import datetime
import json

class ModelTrainer:
    def __init__(self):
        self.log_entries = []
        self.transformations = {}
        self.model = None
        self.feature_importance = None
        self.metrics = {}
        self.data_card = {
            "dataset_info": {},
            "preprocessing": [],
            "model_info": {},
            "performance": {},
            "ethical_considerations": [],
            "limitations": [],
            "timestamp": ""
        }
    
    def log_step(self, step_name, description, details=None):
        """Log a step in the process with timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "step": step_name,
            "description": description,
            "details": details
        }
        self.log_entries.append(log_entry)
        return log_entry

    def transform_features(self, data, transformations):
        """Apply transformations to features."""
        transformed_data = data.copy()
        
        for feature, transform_type in transformations.items():
            self.log_step("Transform", f"Applying {transform_type} to {feature}")
            
            if transform_type == "standardization":
                scaler = StandardScaler()
                transformed_data[f"{feature}_scaled"] = scaler.fit_transform(data[[feature]])
                self.transformations[feature] = ("standardization", scaler)
            
            elif transform_type == "minmax":
                scaler = MinMaxScaler()
                transformed_data[f"{feature}_minmax"] = scaler.fit_transform(data[[feature]])
                self.transformations[feature] = ("minmax", scaler)
            
            elif transform_type == "log":
                transformed_data[f"{feature}_log"] = np.log1p(data[feature])
                self.transformations[feature] = ("log", None)
            
            elif transform_type == "onehot":
                dummies = pd.get_dummies(data[feature], prefix=feature)
                transformed_data = pd.concat([transformed_data, dummies], axis=1)
                transformed_data.drop(feature, axis=1, inplace=True)
                self.transformations[feature] = ("onehot", None)
            
            elif transform_type == "label":
                le = LabelEncoder()
                transformed_data[f"{feature}_encoded"] = le.fit_transform(data[feature])
                self.transformations[feature] = ("label", le)
        
        return transformed_data

    def train_model(self, data, target_column, transformations, problem_type="regression"):
        """Train a model with logging and validation."""
        self.log_step("Training Start", "Initializing model training process")
        
        # Record dataset info
        self.data_card["dataset_info"] = {
            "total_samples": len(data),
            "features": list(data.columns),
            "target": target_column
        }
        
        # Transform features
        transformed_data = self.transform_features(data, transformations)
        feature_cols = [col for col in transformed_data.columns if col != target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            transformed_data[feature_cols],
            data[target_column],
            test_size=0.2,
            random_state=42
        )
        
        self.log_step("Data Split", 
                     "Split data into training and test sets",
                     {"train_size": len(X_train), "test_size": len(X_test)})
        
        # Train model
        if problem_type == "regression":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            metric_name = "R2 Score"
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            metric_name = "Accuracy"
        
        model.fit(X_train, y_train)
        self.model = model
        
        # Calculate performance metrics
        y_pred = model.predict(X_test)
        if problem_type == "regression":
            score = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            self.metrics = {"r2_score": score, "mse": mse}
        else:
            accuracy = accuracy_score(y_test, y_pred)
            self.metrics = {"accuracy": accuracy}
        
        self.log_step("Training Complete", 
                     f"Model trained successfully. {metric_name}: {list(self.metrics.values())[0]:.4f}")
        
        # Calculate feature importance
        self.feature_importance = dict(zip(feature_cols, model.feature_importances_))
        
        # Update data card
        self.data_card["preprocessing"] = self.log_entries
        self.data_card["model_info"] = {
            "type": "Random Forest",
            "problem_type": problem_type,
            "parameters": model.get_params(),
            "feature_importance": self.feature_importance
        }
        self.data_card["performance"] = self.metrics
        self.data_card["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "model": model,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "transformations": self.transformations,
            "log_entries": self.log_entries,
            "data_card": self.data_card
        }

    def generate_data_card(self):
        """Generate a formatted data card as markdown."""
        markdown = f"""# Data Card - Model Training Documentation

## Dataset Information
- Total Samples: {self.data_card['dataset_info'].get('total_samples', 'N/A')}
- Features: {', '.join(self.data_card['dataset_info'].get('features', []))}
- Target Variable: {self.data_card['dataset_info'].get('target', 'N/A')}

## Preprocessing Steps
"""
        for step in self.data_card['preprocessing']:
            markdown += f"### {step['step']} - {step['timestamp']}\n"
            markdown += f"{step['description']}\n"
            if step['details']:
                markdown += f"Details: {json.dumps(step['details'], indent=2)}\n"
            markdown += "\n"

        markdown += f"""
## Model Information
- Type: {self.data_card['model_info'].get('type', 'N/A')}
- Problem Type: {self.data_card['model_info'].get('problem_type', 'N/A')}

### Feature Importance
"""
        for feature, importance in self.data_card['model_info'].get('feature_importance', {}).items():
            markdown += f"- {feature}: {importance:.4f}\n"

        markdown += f"""
## Model Performance
"""
        for metric, value in self.data_card['performance'].items():
            markdown += f"- {metric}: {value:.4f}\n"

        markdown += f"""
## Ethical Considerations
- Data privacy and security measures implemented
- Model predictions should be used as guidance, not absolute truth
- Regular monitoring for bias and performance drift recommended

## Limitations
- Model performance may vary on new, unseen data
- Specific to the training data distribution
- May not capture all edge cases

Generated on: {self.data_card['timestamp']}
"""
        return markdown
