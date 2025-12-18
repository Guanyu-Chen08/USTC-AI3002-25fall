import numpy as np
import pandas as pd
from basemodel import LinearModel
from basetrainer import Trainer
from sklearn.preprocessing import PolynomialFeatures

MEANS = np.array([7.56955126e+01, 7.58841184e+01, 2.55991265e+01, 1.44647190e+01,
       1.44951189e+01, 1.73778796e+01, 1.73946024e+01, 5.00415915e+00,
       2.34273872e+00, 2.34712774e+00, 5.00448959e-01, 4.99551041e-01,
       4.85895511e-01, 4.85967345e-01, 1.43684182e+05, 1.92616963e+02,
       8.63165583e+02, 6.62865183e-02, 6.39629771e+00, 6.40871770e+00,
       3.89626222e+03, 5.53222806e+00, 1.29547764e+06, 7.49247091e+03,
       5.55783758e+03, 1.94581037e+03, 1.15589027e+03, 1.11480846e+03,
       1.38458484e+03, 1.32454085e+03, 3.79003383e+02, 2.05285962e+02,
       1.73160569e+02, 3.78980397e+01, 3.78218388e+01, 3.61384086e+01,
       3.63604600e+01, 1.39671805e+07, 1.54605088e+04, 7.88617111e+04,
       3.59406943e+00, 6.19005912e+02, 4.58973501e+02, 3.36734358e+05,
       4.63422459e+02, 1.29715526e+08, 7.52204650e+03, 1.95045185e+03,
       1.11617938e+03, 1.16040761e+03, 1.32732592e+03, 1.39117558e+03,
       3.80112261e+02, 1.73183096e+02, 2.06110609e+02, 3.79771139e+01,
       3.79476909e+01, 3.64353966e+01, 3.62074836e+01, 1.39868463e+07,
       1.55048272e+04, 7.90306063e+04, 3.61165784e+00, 4.58525260e+02,
       6.21397677e+02, 3.37482348e+05, 4.63855759e+02, 1.30057982e+08,
       7.16758072e+02, 3.80086746e+02, 3.81097241e+02, 4.42007370e+02,
       4.42476299e+02, 1.28067351e+02, 5.98790613e+01, 5.99259541e+01,
       1.28092177e+01, 1.27885296e+01, 1.24478669e+01, 1.24679803e+01,
       4.05122790e+06, 5.17723502e+03, 2.35566930e+04, 1.76291834e+00,
       1.61145772e+02, 1.61338630e+02, 1.09411816e+05, 1.39753094e+02,
       3.72088951e+07, 2.74826099e+02, 1.92616963e+02, 2.67554029e+02,
       2.58407626e+02, 7.23284223e+01, 3.20629261e+01, 3.45660041e+01,
       7.24643886e+00, 7.22431417e+00, 7.08857777e+00, 7.08386550e+00,
       2.31688433e+06, 3.36366615e+03, 1.16293026e+04, 1.06674939e+00,
       7.56955126e+01, 1.01520771e+02, 5.99296752e+04, 7.81223898e+01,
       1.64466943e+07, 2.76071517e+02, 2.58554281e+02, 2.68290983e+02,
       7.24330118e+01, 3.46880634e+01, 3.22445784e+01, 7.25425433e+00,
       7.24385286e+00, 7.09133617e+00, 7.10547299e+00, 2.31884155e+06,
       3.37344928e+03, 1.16250087e+04, 1.07193756e+00, 1.01470832e+02,
       7.58841184e+01, 6.00509420e+04, 7.82177270e+01, 1.63891064e+07,
       3.89668058e+02, 3.10948761e+02, 8.68402641e+01, 3.72814217e+01,
       4.04584695e+01, 8.69253148e+00, 8.67121133e+00, 8.46099805e+00,
       8.43789643e+00, 2.63283422e+06, 3.67073154e+03, 1.49196302e+04,
       1.14117643e+00, 1.12649949e+02, 1.09954874e+02, 6.91596585e+04,
       9.60771203e+01, 2.22944116e+07, 3.90361796e+02, 8.70595571e+01,
       4.04052553e+01, 3.74036534e+01, 8.70753030e+00, 8.69965735e+00,
       8.45743512e+00, 8.45151604e+00, 2.63778340e+06, 3.67444619e+03,
       1.49432936e+04, 1.14249907e+00, 1.09733800e+02, 1.13155828e+02,
       6.92624182e+04, 9.62340045e+01, 2.23367630e+07, 3.40415915e+01,
       1.17727191e+01, 1.17737391e+01, 2.50004669e+00, 2.50238846e+00,
       2.42680535e+00, 2.42660422e+00, 7.20156189e+05, 9.61991107e+02,
       4.34095076e+03, 3.30556448e-01, 3.20512029e+01, 3.21540108e+01,
       1.95115984e+04, 2.77469740e+01, 6.56969944e+06, 8.96234493e+00,
       5.40174986e+00, 1.17285990e+00, 1.16978543e+00, 1.12573719e+00,
       1.13068651e+00, 3.80403624e+05, 4.37656909e+02, 2.31331375e+03,
       1.22762446e-01, 1.80376335e+01, 1.42053430e+01, 9.71677514e+03,
       1.35294553e+01, 3.69992130e+06, 8.97477929e+00, 1.17538844e+00,
       1.17173212e+00, 1.13033453e+00, 1.12848123e+00, 3.80606083e+05,
       4.37738282e+02, 2.31957532e+03, 1.22877435e-01, 1.42506627e+01,
       1.80597798e+01, 9.72643139e+03, 1.35368398e+01, 3.71820215e+06,
       5.00448959e-01, 2.49922779e-01, 2.43335656e-01, 2.42559855e-01,
       7.19640934e+04, 9.64424938e+01, 4.32391463e+02, 3.32144752e-02,
       3.20020688e+00, 3.20973917e+00, 1.94982845e+03, 2.77013311e+00,
       6.51870725e+05, 4.99551041e-01, 2.42523938e-01, 2.43177622e-01,
       7.19214890e+04, 9.62484861e+01, 4.31950923e+02, 3.31207551e-02,
       3.19537249e+00, 3.20371953e+00, 1.94781665e+03, 2.76578719e+00,
       6.50211382e+05, 4.85895511e-01, 2.40031319e-01, 6.73842328e+04,
       9.48012442e+01, 3.87741543e+02, 3.29307894e-02, 3.01711072e+00,
       3.04725920e+00, 1.86804916e+03, 2.64902199e+00, 4.96318304e+05,
       4.85967345e-01, 6.74007979e+04, 9.47750393e+01, 3.87354792e+02,
       3.29124943e-02, 3.04637565e+00, 3.01617688e+00, 1.87103192e+03,
       2.65487641e+00, 4.95408287e+05, 3.80383528e+10, 3.32468138e+07,
       1.91752450e+08, 5.17723502e+03, 1.09458551e+06, 1.09750942e+06,
       7.90246361e+08, 8.88791691e+05, 3.64925161e+11, 5.35654149e+04,
       1.43684182e+05, 1.59355654e+01, 1.11480846e+03, 1.11617938e+03,
       8.35243282e+05, 9.65085115e+02, 1.91752450e+08, 1.29547764e+06,
       2.55991265e+01, 7.06717670e+03, 7.10022536e+03, 4.31736319e+06,
       5.64501116e+03, 2.73259392e+09, 1.18032753e-02, 2.70871644e-01,
       2.71055718e-01, 1.92689428e+02, 2.40634720e-01, 2.35566930e+04,
       6.16693006e+01, 3.49126075e+01, 2.72430996e+04, 4.40383447e+01,
       1.21146998e+07, 6.19280948e+01, 2.72948252e+04, 4.40765672e+01,
       1.22206131e+07, 1.89837433e+07, 2.35092411e+04, 7.46435583e+09,
       5.37721804e+01, 9.06827804e+06, 7.28628456e+12])

SCALES = np.array([4.19840480e+01, 4.19957984e+01, 7.83854543e+00, 8.09925935e+00,
       8.12176365e+00, 9.36361884e+00, 9.36961057e+00, 2.99999712e+00,
       1.86384555e+00, 1.86165804e+00, 4.99999798e-01, 4.99999798e-01,
       4.99801024e-01, 4.99803046e-01, 1.31883314e+05, 1.28312589e+02,
       7.41904854e+02, 8.60777134e-02, 4.55594954e+00, 4.56688430e+00,
       1.95009845e+03, 4.81317287e+00, 2.36812632e+06, 6.78622282e+03,
       4.53094782e+03, 1.28221890e+03, 1.03378798e+03, 9.68347015e+02,
       1.18124317e+03, 1.10160658e+03, 3.34290698e+02, 2.40043561e+02,
       1.76901973e+02, 4.81276199e+01, 4.81141041e+01, 4.71967280e+01,
       4.74761584e+01, 1.76526167e+07, 1.51725013e+04, 9.19067600e+04,
       3.50934343e+00, 6.59599110e+02, 4.21837253e+02, 2.95854332e+05,
       5.51344275e+02, 2.75379024e+08, 6.79211610e+03, 1.28326215e+03,
       9.67670504e+02, 1.03617183e+03, 1.10274621e+03, 1.18719988e+03,
       3.34946230e+02, 1.76784485e+02, 2.40001773e+02, 4.81875595e+01,
       4.82167599e+01, 4.75169193e+01, 4.72417384e+01, 1.76520491e+07,
       1.51959162e+04, 9.21233061e+04, 3.53211676e+00, 4.19713754e+02,
       6.61076337e+02, 2.95960427e+05, 5.50640056e+02, 2.76262833e+08,
       3.76250181e+02, 2.66293353e+02, 2.67358899e+02, 2.81338327e+02,
       2.81509765e+02, 8.93401257e+01, 5.28493016e+01, 5.26971810e+01,
       1.39476351e+01, 1.39475015e+01, 1.39201192e+01, 1.39354670e+01,
       4.33560255e+06, 4.27365224e+03, 2.38290711e+04, 2.55299694e+00,
       1.27427480e+02, 1.27489558e+02, 7.40160307e+04, 1.33585740e+02,
       7.48592590e+07, 3.20815872e+02, 1.28312589e+02, 2.46145273e+02,
       2.24889411e+02, 6.40841922e+01, 2.83902837e+01, 3.79375075e+01,
       9.23609006e+00, 9.22312445e+00, 9.24970084e+00, 9.24147806e+00,
       3.06025341e+06, 4.06791309e+03, 1.16791613e+04, 1.85115031e+00,
       4.19840480e+01, 1.11960464e+02, 5.34206713e+04, 9.18927542e+01,
       3.02488726e+07, 3.21909360e+02, 2.24951827e+02, 2.46400334e+02,
       6.41166200e+01, 3.83864657e+01, 2.86063994e+01, 9.25412833e+00,
       9.25007911e+00, 9.24877932e+00, 9.26753603e+00, 3.06311040e+06,
       4.08009123e+03, 1.16338168e+04, 1.85745859e+00, 1.12081670e+02,
       4.19957984e+01, 5.35443134e+04, 9.21223423e+01, 2.98591877e+07,
       3.86805586e+02, 2.61637455e+02, 7.53137515e+01, 3.06430967e+01,
       4.18994475e+01, 1.09166393e+01, 1.09156758e+01, 1.08801493e+01,
       1.08568246e+01, 3.15941949e+06, 4.01280465e+03, 1.66528863e+04,
       1.73987642e+00, 1.10658270e+02, 1.05705942e+02, 5.56679226e+04,
       1.07910908e+02, 4.82637604e+07, 3.87118601e+02, 7.55998202e+01,
       4.17827260e+01, 3.07131407e+01, 1.09344244e+01, 1.09409625e+01,
       1.08756931e+01, 1.08697883e+01, 3.17236364e+06, 4.01025021e+03,
       1.67016738e+04, 1.73897237e+00, 1.05408269e+02, 1.11553186e+02,
       5.58080357e+04, 1.08185786e+02, 4.83098022e+07, 2.99999712e+01,
       1.30415566e+01, 1.30242886e+01, 3.27765924e+00, 3.28163373e+00,
       3.25643073e+00, 3.25609531e+00, 8.84217043e+05, 9.41992976e+02,
       5.11131087e+03, 5.37428594e-01, 3.28464198e+01, 3.30066889e+01,
       1.63392433e+04, 3.26584479e+01, 1.47777916e+07, 1.57474776e+01,
       6.70260005e+00, 1.76408956e+00, 1.76253471e+00, 1.72544836e+00,
       1.73585886e+00, 5.72011660e+05, 4.58865307e+02, 3.38481873e+03,
       1.42006040e-01, 2.45552127e+01, 1.52344676e+01, 1.05684932e+04,
       1.92510828e+01, 9.51137703e+06, 1.57152001e+01, 1.76729012e+00,
       1.76141856e+00, 1.73261933e+00, 1.72809595e+00, 5.70088041e+05,
       4.55357425e+02, 3.39343405e+03, 1.41365634e-01, 1.53807547e+01,
       2.44576655e+01, 1.05388973e+04, 1.91876760e+01, 9.56584711e+06,
       4.99999798e-01, 4.32968109e-01, 4.29096043e-01, 4.28631044e-01,
       1.17957364e+05, 1.32231787e+02, 6.81841879e+02, 6.94556349e-02,
       4.53954397e+00, 4.55279221e+00, 2.38779857e+03, 4.38448217e+00,
       1.81712638e+06, 4.99999798e-01, 4.28609470e-01, 4.29001476e-01,
       1.17996319e+05, 1.32372178e+02, 6.80903651e+02, 6.94621864e-02,
       4.53756177e+00, 4.55141654e+00, 2.38888764e+03, 4.38189791e+00,
       1.80589152e+06, 4.99801024e-01, 4.27102195e-01, 1.12944063e+05,
       1.32707084e+02, 5.88196225e+02, 6.88271365e-02, 4.38150835e+00,
       4.43285306e+00, 2.34362773e+03, 4.34493161e+00, 1.10585741e+06,
       4.99803046e-01, 1.13002301e+05, 1.32233529e+02, 5.87677251e+02,
       6.87430491e-02, 4.43985813e+00, 4.37697315e+00, 2.34627647e+03,
       4.35680266e+00, 1.10492524e+06, 6.92075436e+10, 4.85238404e+07,
       3.15621029e+08, 4.27365224e+03, 1.49122493e+06, 1.49782280e+06,
       1.09131841e+09, 1.32794169e+06, 9.76354826e+11, 9.52251798e+04,
       1.31883314e+05, 3.44722934e+01, 9.68347015e+02, 9.67670504e+02,
       8.57380265e+05, 9.88432597e+02, 3.15621029e+08, 2.36812632e+06,
       7.83854543e+00, 9.61703496e+03, 9.72424133e+03, 5.37744706e+06,
       7.84663671e+03, 8.58173714e+09, 5.02413647e-02, 2.25426083e-01,
       2.25424400e-01, 1.78089904e+02, 2.40826958e-01, 2.38290711e+04,
       8.25501572e+01, 2.87012512e+01, 2.69160436e+04, 6.55473538e+01,
       2.90429257e+07, 8.27952418e+01, 2.69806919e+04, 6.54947509e+01,
       2.95903633e+07, 1.77646323e+07, 2.61211957e+04, 1.68654888e+10,
       6.94432117e+01, 2.06464975e+07, 3.39311554e+13])

def load_and_preprocess_data(data_file: str = "data/train.csv"):
    dataset = pd.read_csv(data_file)
    """
    Divide the dataset into features and target

    You can do all possible modifications to features, but DO NOT change the targets

    return:
        features (np.ndarray): Input features, shape [num_samples, in_features]
        targets (np.ndarray): Target values, shape [num_samples]
    """
    # raise NotImplementedError("Not Implemented Yet.")
    original_features = dataset.drop('Run_time', axis=1)
    targets = dataset['Run_time'].to_numpy(dtype=np.float32)

    features_dict = {}
    features_dict['total_flops'] = dataset['MWG'] * dataset['NWG'] * dataset['KWG']
    total_threads = dataset['MDIMC'] * dataset['NDIMC']
    features_dict['total_threads'] = total_threads
    features_dict['work_per_thread'] = features_dict['total_flops'] / (total_threads)
    features_dict['thread_utilization'] = total_threads / (dataset['MWG'] * dataset['NWG'])
    features_dict['work_per_thread_m'] = dataset['MWG'] / (dataset['MDIMC'])
    features_dict['work_per_thread_n'] = dataset['NWG'] / (dataset['NDIMC'])
    features_dict['global_memory_loads'] = (dataset['MWG'] * dataset['KWG'] + 
                                            dataset['NWG'] * dataset['KWG'])
    features_dict['work_variance'] = np.abs(features_dict['work_per_thread_m'] - 
                                            features_dict['work_per_thread_n'])
    features_dict['work_squared'] = features_dict['work_per_thread'] ** 2
    
    handcrafted = pd.DataFrame(features_dict)

    combined_for_poly = pd.concat([
        original_features,
        handcrafted
    ], axis=1)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    features = poly.fit_transform(combined_for_poly)

    features = (features - MEANS) / SCALES
    features = features.astype(np.float32)

    print(f"Data size: {features.shape[0]}. Features num: {features.shape[1]}")
    
    return features, targets

class LinearRegressionModel(LinearModel):
    def __init__(self, in_features: int, out_features: int):
        """
        Linear regression model, inherits from LinearModel.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features (usually 1).
        """
        # raise NotImplementedError("Not Implemented Yet.")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(in_features, out_features) * 0.001
        self.bias = np.zeros(out_features)

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the output given input.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
        """
        # raise NotImplementedError("Not Implemented Yet.")

        return features @ self.weight + self.bias

    def gradient(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> tuple:
        """
        Compute gradients for MSE loss.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True values, shape [batch_size, out_features].
            predictions (np.ndarray): Predicted values, shape [batch_size, out_features].

        Returns:
            tuple: (dw, db), gradients for weights and bias.
        """
        # raise NotImplementedError("Not Implemented Yet.")
        lambda_reg = 0.01
        batch_size = features.shape[0]
        errors = predictions - targets
        dw = (2 / batch_size) * (features.T @ errors + lambda_reg * self.weight) 
        db = (2 / batch_size) * np.sum(errors, axis=0)

        return dw, db

    def backpropagation(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray, learning_rate: float = 0.01) -> float:
        """
        Perform backpropagation, compute MSE loss and update parameters.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True values, shape [batch_size, out_features].
            predictions (np.ndarray): True values, shape [batch_size, out_features].
            learning_rate (float): Learning rate, default 0.01.
        """
        # raise NotImplementedError("Not Implemented Yet.")
        dw, db = self.gradient(features, targets, predictions)
        self.weight -= dw * learning_rate
        self.bias -= db * learning_rate
        loss = np.mean((targets - predictions) ** 2)

        return loss

class LinearRegressionTrainer(Trainer):
    def __init__(self, model, train_dataloader, eval_dataloader=None, 
                 save_dir=None, learning_rate=0.01, eval_strategy="epoch", 
                 eval_steps=100, num_epochs=10, eval_metric="mae"):
        super().__init__(model, train_dataloader, eval_dataloader, save_dir, 
                         learning_rate, eval_strategy, eval_steps, num_epochs, eval_metric)

    def compute_loss(self, batch_pred, batch_grd):
        """
        Compute loss based on model type with detailed checks for linear regression.

        Args:
            batch_pred: Predicted values, shape [batch_size, out_features].
            batch_grd: True values/labels, shape [batch_size, out_features].

        Returns:
            float: Mean loss for the batch.
        """
        # raise NotImplementedError("Not Implemented Yet.")
        loss = np.mean((batch_grd - batch_pred) ** 2)

        return loss

def linear_regression_analytic(X, y):
    """
    Calculate the analytical linear regression results.

    Args:
        X (np.ndarray): Input features, shape [num_samples, in_features]
        y (np.ndarray): True values, shape [num_samples, out_features]

    Return:
        weight (np.ndarray): Model weight
        bias (np.ndarray | float): Model bias
    """
    # raise NotImplementedError("Not Implemented Yet.")
    y = y.reshape(-1, 1)
    in_features = X.shape[1]
    out_features = y.shape[1]
    X_Augmented = np.c_[X, np.ones((X.shape[0], 1))]
    W_Augmented = np.linalg.pinv(X_Augmented.T @ X_Augmented) @ X_Augmented.T @ y
    weight = W_Augmented[:in_features]
    bias = W_Augmented[-out_features:]

    return weight, bias

class LogisticRegressionModel(LinearModel):
    def __init__(self, in_features: int, out_features: int):
        """
        Logistic regression model, inherits from LinearModel.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features (usually 1 for binary classification).
        """
        # raise NotImplementedError("Not Implemented Yet.")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(in_features, out_features) * 0.001
        self.bias = np.zeros(out_features)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid function.

        Args:
            x (np.ndarray): Input values.

        Returns:
            np.ndarray: Sigmoid output.
        """
        # raise NotImplementedError("Not Implemented Yet.")

        return 1 / (1 + np.exp(-x))

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the output given input.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
        """
        # raise NotImplementedError("Not Implemented Yet.")
        score = features @ self.weight + self.bias

        return self._sigmoid(score)

    def gradient(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> tuple:
        """
        Compute gradients for binary cross-entropy loss.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True labels (0 or 1), shape [batch_size, out_features].
            predictions (np.ndarray): Predicted probabilities, shape [batch_size, out_features].

        Returns:
            tuple: (dw, db), gradients for weights and bias.
        """
        # raise NotImplementedError("Not Implemented Yet.")
        lambda_reg = 0.01
        batch_size = features.shape[0]
        errors = predictions - targets
        dw = (1 / batch_size) * (features.T @ errors + lambda_reg * self.weight) 
        db = (1 / batch_size) * np.sum(errors, axis=0)

        return dw, db
    
    def backpropagation(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray, learning_rate: float = 0.01) -> float:
        """
        Perform backpropagation, compute binary cross-entropy loss and update parameters.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True labels (0 or 1), shape [batch_size, out_features].
            learning_rate (float): Learning rate, default 0.01.

        Returns:
            float: Binary cross-entropy loss for the batch.
        """
        # raise NotImplementedError("Not Implemented Yet.")
        epsilon = 1e-9
        dw, db = self.gradient(features, targets, predictions)
        self.weight -= dw * learning_rate
        self.bias -= db * learning_rate
        loss = -np.mean(targets * np.log(predictions + epsilon) + (1 - targets) * np.log(1 - predictions + epsilon))

        return loss

class LogisticRegressionTrainer(Trainer):
    def __init__(self, model, train_dataloader, eval_dataloader=None, 
                 save_dir=None, learning_rate=0.01, eval_strategy="epoch", 
                 eval_steps=100, num_epochs=10, eval_metric="f1"):
        super().__init__(model, train_dataloader, eval_dataloader, save_dir, 
                         learning_rate, eval_strategy, eval_steps, num_epochs, eval_metric)
        
    def compute_loss(self, batch_pred, batch_grd):
        # raise NotImplementedError("Not Implemented Yet.")
        epsilon = 1e-9
        loss = -np.mean(batch_grd * np.log(batch_pred + epsilon) + (1 - batch_grd) * np.log(1 - batch_pred + epsilon))

        return loss