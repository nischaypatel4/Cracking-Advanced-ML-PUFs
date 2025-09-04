import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################
	
	# Use this method to train your models using training CRPs
	# X_train has 8 columns containing the challenge bits
	# y_train contains the values for responses
	
	# THE RETURNED MODEL SHOULD BE ONE VECTOR AND ONE BIAS TERM
	# If you do not wish to use a bias term, set it to 0

    X_train_mapped = my_map(X_train)
    model = LinearSVC(C=10, max_iter=1000)
    # model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000, fit_intercept=False, C = 100.0, random_state=42)
    model.fit(X_train_mapped, y_train)
    w = model.coef_.flatten()
    b = 0.0
    return w, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
    X_bin = 1 - 2 * X  # {0,1} -> {1,-1}
    m, n = X_bin.shape
    feat1 = X_bin
    # Quadratic
    # arbiter = np.cumprod(np.hstack([np.ones((m, 1)), X_bin]), axis=1)
    # feat2 = khatri_rao(arbiter.T, arbiter.T).T
    
    # majority_bit = (X.sum(axis=1) > n // 2).astype(int)
    # majority_feat = (1 - 2 * majority_bit)[:, None]  # {0,1} → {+1,-1}
    # global_parity = (X.sum(axis=1) % 2)
    # parity_feat = (1 - 2 * global_parity)[:, None]  # {0,1} → {+1,-1}
    # weights = np.arange(1, n + 1)
    # weighted_sum = (X * weights).sum(axis=1, keepdims=True)
    # weighted_feat = (weighted_sum / weights.sum()) * 2 - 1  # scaled to [-1, 1]
    quad_feats = [X_bin[:, i:i+1] * X_bin[:, i:i+1] for i in range(n)]  # squares (redundant if X_bin ∈ {±1})
    quad_feats += [X_bin[:, i:i+1] * X_bin[:, i+1:i+2] for i in range(n - 1)]  # adjacent pairs
    feat_quad = np.hstack(quad_feats)
    pairwise_triplet_feats = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Use thresholding to reduce dimensionality, simplifying cubic interactions
                triplet_product = (X_bin[:, i] * X_bin[:, j] * X_bin[:, k])
                pairwise_triplet_feats.append(triplet_product[:, None])  # Add as a column vector
    feat_triplet = np.hstack(pairwise_triplet_feats)


    # Cubic
    # cubic_feats = []
    # for i in range(n):
    #     for j in range(i+1, n):
    #         for k in range(j+1, n):
    #             cubic_feats.append((X_bin[:, i] * X_bin[:, j] * X_bin[:, k])[:, None])
    # feat3 = np.hstack(cubic_feats)

    X_int = X.astype(int)
    xor_feats = []
    for i in range(n):
        for j in range(i + 1, n):
            xor_val = X_int[:, i] ^ X_int[:, j]  # still {0,1}
            xor_feats.append((1 - 2 * xor_val)[:, None])  # map to {1,-1}
    feat_xor = np.hstack(xor_feats)

    # Optionally map XOR features to ±1 space for consistency
    feat_xor = 1 - 2 * feat_xor  # {0,1} → {1,-1}

    
    feat = np.hstack([feat_triplet, feat_xor, feat_quad])
    return feat


################################
# Non Editable Region Starting #
################################
def my_decode( w ):
################################
#  Non Editable Region Ending  #
################################


	# Use this method to invert a PUF linear model to get back delays
	# w is a single 65-dim vector (last dimension being the bias term)
	# The output should be four 64-dimensional vectors
	b = w[-1]
	w_vector = np.array(w[:-1])
	p = np.zeros(64)
	q = np.zeros(64)
	r = np.zeros(64)
	s = np.zeros(64)
	alpha = np.zeros(64)
	alpha[0] = w_vector[0]
	alpha[1:] = w_vector[1:]
	for i in range(64):
		if i < 63:
			beta_i = 0.0
		else:
			beta_i = b
		D = alpha[i] + beta_i  # p_i - q_i
		E = alpha[i] - beta_i  # r_i - s_i
		p[i] = max(D, 0)
		q[i] = max(-D, 0)
		r[i] = max(E, 0)
		s[i] = max(-E, 0)
	
	return p, q, r, s

