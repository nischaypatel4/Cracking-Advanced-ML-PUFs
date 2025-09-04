import numpy as np
import sklearn
from scipy.linalg import khatri_rao

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

	D = X_train.shape[1]
	
	return np.zeros( D, ), 0


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	return np.cumprod( np.flip( 2 * X - 1 , axis = 1 ), axis = 1 )


################################
# Non Editable Region Starting #
################################
def my_decode( w ):
################################
#  Non Editable Region Ending  #
################################

	d = w.shape[0] - 1		# w is 65 dimensional whereas the delay vectors should be 64 dimensional
	p = np.zeros( d, )		# delays must be non-negative but zeros are allowed
	q = np.zeros_like( p )
	r = np.zeros_like( p )
	s = np.zeros_like( p )
	
	return p, q, r, s
