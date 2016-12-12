import matplotlib.pyplot as plt
import numpy as np
import pickle as cucumber

title = 'ASDFASDFASDF';
output_filename = 'asdf'

training_error_file = 'pickles/train_error_online_NN3.pickle'
test_error_file = 'pickles/test_error_online_NN3.pickle'

with open(training_error_file,'r') as f:
  train_error = cucumber.load(f);

with open(test_error_file,'r') as f:
  test_error = cucumber.load(f);

train_error_fixed = train_error[0];
test_error_fixed = test_error[0]
for i in range(1,len(train_error)):
  if (i%59):
    train_error_fixed = np.append(train_error_fixed, train_error[i]);
    test_error_fixed = np.append(test_error_fixed, test_error[i]);

x_vals = range(0,len(train_error_fixed));

plt.plot(x_vals, train_error_fixed)
plt.plot(x_vals, test_error_fixed)

axes = plt.gca()
axes.set_xlim([0,len(x_vals)])
axes.set_ylim([0,100])

plt.xlabel('Training Examples (x1000)')
plt.ylabel('Error')
plt.title(title)
plt.grid(False)
plt.savefig("outputPlots/" + output_filename)
plt.show()