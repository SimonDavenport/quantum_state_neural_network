bin_PROGRAMS = \
line_search_test \
bfgs_test \
linear_algebra_test \
single_layer_perceptron_test

##  line search test
line_search_test_SOURCES = \
test/line_search_test.cpp \
utilities/optimization/line_search.cpp \
utilities/linear_algebra/dense_vector.cpp \
utilities/linear_algebra/dense_matrix.cpp

line_search_test_LINK = $(FC) $(LDFLAGS) -o line_search_test

##  bfgs test
bfgs_test_SOURCES = \
test/bfgs_test.cpp \
utilities/optimization/bfgs.cpp \
utilities/optimization/lbfgs.cpp \
utilities/optimization/line_search.cpp \
utilities/linear_algebra/dense_vector.cpp \
utilities/linear_algebra/dense_matrix.cpp

bfgs_test_LINK = $(FC) $(LDFLAGS) -o bfgs_test

##  linear algebra test
linear_algebra_test_SOURCES = \
test/linear_algebra_test.cpp \
utilities/linear_algebra/dense_vector.cpp \
utilities/linear_algebra/dense_matrix.cpp

linear_algebra_test_LINK = $(FC) $(LDFLAGS) -o linear_algebra_test

##  single_layer_perceptron_test
single_layer_perceptron_test_SOURCES = \
test/single_layer_perceptron_test.cpp \
neural_network/single_layer_perceptron.cpp \
neural_network/loss_function_data.cpp \
neural_network/network_functions.cpp \
utilities/optimization/bfgs.cpp \
utilities/optimization/lbfgs.cpp \
utilities/optimization/line_search.cpp \
utilities/linear_algebra/dense_vector.cpp \
utilities/linear_algebra/dense_matrix.cpp

single_layer_perceptron_test_LINK = $(FC) $(LDFLAGS) -o single_layer_perceptron_test

