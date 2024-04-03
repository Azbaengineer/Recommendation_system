
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


import sys
# import Q Widgets
from PyQt6.QtWidgets import (
    QWidget,
    QSlider,
    QLabel,
    QPushButton, QTabWidget, QComboBox, QApplication, QVBoxLayout, QSizePolicy, QMainWindow, QMessageBox, QSpinBox,
    QDoubleSpinBox,

)


# Custom Logistic RegressionCV
class LogisticRegressionCVCustom(LogisticRegressionCV):
    def fit(self, X, y, **fit_params):
        feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.classes_ = pd.unique(y)
        return super().fit(X, y, **fit_params)


    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)

# Load and preprocess the dataset
data = pd.read_csv('framingham.csv')
print(data.head())
print(data.describe())
print(data.info())
print(data.isnull().sum())
data.drop(['education'], axis=1, inplace=True)
data.rename(columns={'male': 'Sex_male'}, inplace=True)
data.dropna(inplace=True)
#correlation
corr = data.corr()
# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
print(corr)

# Select features
features = ['age', 'sysBP', 'prevalentHyp', 'diaBP', 'glucose', 'Sex_male', 'diabetes']
X = data[features]
y = data['TenYearCHD']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2, stratify=y)

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Create and fit the model
model = LogisticRegressionCVCustom(max_iter=1000)
model.fit(x_train_scaled, y_train)

# Make predictions and evaluate accuracy
y_test_predict = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test_predict, y_test)
print("Accuracy on test data: ", accuracy)

# Create a violin plot for 'Sex_male' and 'age'
plt.figure(figsize=(12, 6))
sns.violinplot(x='diaBP', y='diabetes', data=data, hue='TenYearCHD', split=True)
plt.title('Violin Plot for diaBP and diabetes')
plt.show()



class CustomPlotCanvas(FigureCanvas):
    def __init__(self, parent, x_data, y_data, title='Plot', plot_type='scatterplot', color=None):
        fig, ax = plt.subplots(figsize=(8.9, 5))
        super().__init__(fig)
        self.setParent(parent)

        if plot_type == 'scatterplot':
            sns.scatterplot(x=x_data, y=y_data, ax=ax, color=color)
        elif plot_type == 'regplot':
            sns.regplot(x=x_data, y=y_data, ax=ax, color=color)
        elif plot_type == 'barplot':
            sns.barplot(x=x_data.index, y=y_data, ax=ax, color=color)
        elif plot_type == 'lineplot':
            sns.lineplot(x=x_data.index, y=y_data, ax=ax, color=color)
        else:
            raise ValueError(f'Invalid plot_type: {plot_type}')

        ax.set_title(title)
class MainWindow(QTabWidget):
    update_label_signal = pyqtSignal(int)
    def __init__(self):
        super().__init__()

        self.label_age = QLabel("Age:")

        self.font = QFont("Arial", 12)

        self.label_age.setFont(self.font)

        # Create a label for the prediction
        self.label_prediction = QLabel("", self)
        self.label_prediction.setGeometry(900, 10, 300, 50)
        self.label_prediction.setFont(self.font)

        self.update_label_signal.connect(self.update_label_text)
        # Set the window title
        self.setWindowTitle("Heart disease")

        # Set the window dimensions
        self.setGeometry(100, 100, 800, 600)

        # Create a label for age
        self.label_age = QLabel("Age:", self)
        self.label_age.move(950, 140)
        self.label_age.setFont(self.font)

        # Create a slider for age
        self.value_age = QSlider(Qt.Orientation.Horizontal, self)
        self.value_age.setMinimum(20)
        self.value_age.setMaximum(100)
        self.value_age.setTickInterval(5)
        self.value_age.setSingleStep(5)
        self.value_age.move(1180, 135)
        self.value_age.resize(150, 30)

        # Create a label to display the current value of the age slider
        self.label_age_value = QLabel(str(self.value_age.value()), self)
        self.label_age_value.setFont(self.font)
        self.label_age_value.setGeometry(1260, 135, 50, 30)

        # Connect the slider's valueChanged signal to update the label
        self.value_age.valueChanged.connect(self.update_age_label)

        # Apply a stylesheet to change the style of the age slider
        self.value_age.setStyleSheet("""
                  QSlider::handle:horizontal {
                      background: purple;  /* Handle color */
                      border: 1px solid #5c5c5c; /* Handle border */
                      width: 18px; /* Handle width */
                      margin: -2px 0; /* Handle margin */
                      border-radius: 9px; /* Handle border-radius */
                  }
                  QSlider::add-page:horizontal {
                      background: lightgray;  /* Unfilled part color */
                      height: 8px; /* Unfilled part height */
                      border-radius: 4px; /* Unfilled part border-radius */
                  }
                  QSlider::sub-page:horizontal {
                      background: teal;  /* Filled part color */
                      height: 8px; /* Filled part height */
                      border-radius: 4px; /* Filled part border-radius */
                  }
              """)
        # Create a label for systolic blood pressure
        self.label_sysbp = QLabel("Systolic Blood Pressure :", self)
        self.label_sysbp.move(950, 180)
        self.label_sysbp.setFont(self.font)

        # Create a double spin box for systolic blood pressure
        self.double_spin_sysbp = QDoubleSpinBox(self)
        self.double_spin_sysbp.setMinimum(80.0)
        self.double_spin_sysbp.setMaximum(300.0)
        self.double_spin_sysbp.setSingleStep(1.0)
        self.double_spin_sysbp.setValue(120.0)
        self.double_spin_sysbp.setGeometry(1180, 180, 150, 30)

        # Create a label for sex_male
        self.label_sex_male = QLabel("Sex_Male:", self)
        self.label_sex_male.move(950, 220)
        self.label_sex_male.setFont(self.font)

        # Create a combo box for sex_male (read-only)
        self.value_sex_male = QComboBox(self)
        self.value_sex_male.addItem("Male")
        self.value_sex_male.setCurrentIndex(1)

        self.value_sex_male.move(1180, 220)
        self.value_sex_male.resize(150, 30)

        # Create a label for glucose
        self.label_glucose = QLabel("Glucose Level:", self)
        self.label_glucose.move(950, 260)
        self.label_glucose.setFont(self.font)

        # Create a spin box for glucose
        self.spin_glucose = QSpinBox(self)
        self.spin_glucose.setMinimum(40)
        self.spin_glucose.setMaximum(400)
        self.spin_glucose.setSingleStep(1)
        self.spin_glucose.setValue(100)
        self.spin_glucose.move(1180, 260)
        self.spin_glucose.resize(150, 30)

        # Create a label for diabetes
        self.label_diabetes = QLabel("Diabetes (0/1):", self)
        self.label_diabetes.move(950, 300)
        self.label_diabetes.setFont(self.font)

        # Create a combo box for diabetes
        self.value_diabetes = QComboBox(self)
        self.value_diabetes.addItems(["","No", "Yes"])
        self.value_diabetes.setCurrentIndex(0)
        self.value_diabetes.move(1180, 300)
        self.value_diabetes.resize(150, 30)

        # Create a label for prevalentHyp
        self.label_prevalent_hyp = QLabel("Prevalent Hypertension:", self)
        self.label_prevalent_hyp.move(950, 340)
        self.label_prevalent_hyp.setFont(self.font)

        # Create a combo box for prevalentHyp (read-only)
        self.value_prevalent_hyp = QComboBox(self)
        self.value_prevalent_hyp.addItems(["", "No", "Yes"])
        self.value_prevalent_hyp.setCurrentIndex(0)
        self.value_prevalent_hyp.move(1180, 340)
        self.value_prevalent_hyp.resize(150, 30)

        # Create a label for diaBP
        self.label_diabetes_pressure = QLabel("Diastolic Blood Pressure :", self)
        self.label_diabetes_pressure.move(950, 380)
        self.label_diabetes_pressure.setFont(self.font)

        # Create a spin box for diaBP
        self.spin_diabetes_pressure = QSpinBox(self)
        self.spin_diabetes_pressure.setMinimum(40)
        self.spin_diabetes_pressure.setMaximum(150)
        self.spin_diabetes_pressure.setSingleStep(1)
        self.spin_diabetes_pressure.setValue(80)
        self.spin_diabetes_pressure.move(1180, 380)
        self.spin_diabetes_pressure.resize
        # Connect the signal to the slot
        self.update_label_signal.connect(self.update_label_text)
        # predict button
        self.button = QPushButton("Predict", self)
        self.button.setCheckable(True)
        self.button.move(950, 500)
        self.button.setCheckable(True)
        self.button.clicked.connect(self.predict_chd)
        # output label
        self.prediction = QLabel("", self)
        self.prediction.setGeometry(1180, 550, 300, 50)
        self.prediction.setFont(QFont("Arial", 10, weight=QFont.Weight.Bold))
        self.tab = QWidget(self)
        self.addTab(main, "Heart Prediction")



    def update_age_label(self):
        # Update the label with the current value of the age slider
        self.label_age_value.setText(str(self.value_age.value()))

    def predict_chd(self):
        # Gather input data
        age = self.value_age.value()
        sys_bp = self.double_spin_sysbp.value()
        prevalent_hyp_mapping = {"No": 1, "Yes": 1}
        selected_text_hyp = self.value_prevalent_hyp.currentText()
        prevalent_hyp = prevalent_hyp_mapping.get(selected_text_hyp, None)
        dia_bp = self.spin_diabetes_pressure.value()
        glucose = self.spin_glucose.value()
        sex_male = 1 if self.value_sex_male.currentText() == "Male" else 0
        diabetes_mapping = {"No": 1, "Yes": 1}
        selected_dia = self.value_diabetes.currentText()
        diabetes = diabetes_mapping.get(selected_dia, None)

        # Check if all input fields are filled
        if all([age, sys_bp, prevalent_hyp, dia_bp, glucose, sex_male, diabetes]):
            # Create a DataFrame with the input data
            input_data = pd.DataFrame({
                'age': [int(age)],
                'sysBP': [float(sys_bp)],
                'prevalentHyp': [int(prevalent_hyp)],
                'diaBP': [float(dia_bp)],
                'glucose': [float(glucose)],
                'Sex_male': [int(sex_male)],
                'diabetes': [int(diabetes)]
            }, columns=['age', 'sysBP', 'prevalentHyp', 'diaBP', 'glucose', 'Sex_male', 'diabetes'])

            # Scale input data if needed (use the same scaler as during training)
            input_data_scaled = scaler.transform(input_data)  # Assuming 'scaler' is defined and fitted during training

            # Make the prediction
            prediction = model.predict(input_data_scaled)
            print(f"Prediction: {prediction}")

            # Update the prediction label
            if prediction[0] == 0:
                self.prediction.setText("There is no significant possibility :)")
            else:
                self.prediction.setText("There is a possibility of getting CHD in ten years.")
        else:
            self.prediction.setText("Please fill all the details!")

    def update_label_text(self, prediction):
        # Update the correct label
        self.label_prediction.setText("There is a possibility of getting CHD in ten years." if prediction == 1
                                      else "There is no significant possibility .")

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirmation', 'Are you sure you want to close?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()
class Window(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setGeometry(100, 100, 900, 700)


        # Creating a layout
        self.layout = QVBoxLayout(self)



        # Display statistics labels for "sysBP"
        sysBP_mean_label = QLabel(f'Mean of sysBP: {data["sysBP"].mean():.2f}', self)
        sysBP_median_label = QLabel(f'Median of sysBP: {data["sysBP"].median():.2f}', self)
        sysBP_std_label = QLabel(f'Standard Deviation of sysBP: {data["sysBP"].std():.2f}', self)

        self.layout.addWidget(sysBP_mean_label)
        self.layout.addWidget(sysBP_median_label)
        self.layout.addWidget(sysBP_std_label)


        font = QFont("Arial", 12)
        sysBP_mean_label.setFont(font)
        sysBP_median_label.setFont(font)
        sysBP_std_label.setFont(font)

        # Creating a scatter plot for "sysBP" vs "diaBP" with a custom color
        self.sysBP_diabp_plot = CustomPlotCanvas(
            self, x_data=data["sysBP"], y_data=data["diaBP"], title='sysBP vs diaBP Scatter Plot',
            plot_type='scatterplot', color='purple'
        )
        self.layout.addWidget(self.sysBP_diabp_plot)

        # Display statistics labels for "TenYearCHD"
        tenYearCHD_mean_label = QLabel(f'Mean of TenYearCHD: {data["TenYearCHD"].mean():.2f}', self)
        tenYearCHD_median_label = QLabel(f'Median of TenYearCHD: {data["TenYearCHD"].median():.2f}', self)
        tenYearCHD_std_label = QLabel(f'Standard Deviation of TenYearCHD: {data["TenYearCHD"].std():.2f}', self)


        font = QFont("Arial", 12)
        tenYearCHD_mean_label.setFont(font)
        tenYearCHD_median_label.setFont(font)
        tenYearCHD_std_label.setFont(font)

        self.layout.addWidget(tenYearCHD_mean_label)
        self.layout.addWidget(tenYearCHD_median_label)
        self.layout.addWidget(tenYearCHD_std_label)


        # Creating a bar plot for "TenYearCHD" with a custom color
        self.tenyearchd_plot = CustomPlotCanvas(
            self, x_data=data["TenYearCHD"], y_data=data["TenYearCHD"], title='TenYearCHD Bar Plot', plot_type='barplot', color='teal'
        )
        self.layout.addWidget(self.tenyearchd_plot)


        # Setting the layout for the main window
        self.setLayout(self.layout)

class AppMain(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(900, 800)
        Window(self)




app = QApplication([])
main = AppMain()
window = MainWindow()

window.show()
main.show()
app.exec()
