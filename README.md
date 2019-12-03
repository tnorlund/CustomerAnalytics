
# Customer Analytics

## Campaign Effectiveness

## Customer Churn

*Customer churn* is determined by whether the customer decides to finish their relationship with a company. 

Companies can monitor how often customers leave their business by watching customer churn. The following data is an example of how a telecommunications business can use customer churn to make better business decisions.

### Data

The customer's demographics are recorded in this situation. The recorded demographics are:
- the customer's gender
- whether the customer is a senior citizen
- whether the customer has a partner
- whether the customer has dependents
- how much tenure the customer has with the company

![alt text](https://raw.githubusercontent.com/tnorlund/CustomerAnalytics/master/Demographics.png "Customer Demographics")

The company also offers internet services for its customers. The customer's options for the internet the company offers are whether the customer choses:
- to have the internet service
- to protect themselves using the company's online security
- to backup their devices online

![alt text](https://raw.githubusercontent.com/tnorlund/CustomerAnalytics/master/InternetService.png "Internet Service")

The internet service allows for customers to stream many television series and movies to their devices.

![alt text](https://raw.githubusercontent.com/tnorlund/CustomerAnalytics/master/Streaming.png "Internet Service")

Not only does the company offer an internet service, the company offers a phone service. Each customer can chose to have the phone service, and they also have the option to obtain multiple lines.

![alt text](https://raw.githubusercontent.com/tnorlund/CustomerAnalytics/master/PhoneService.png "Phone Service")

The telecommunications company also allows the customer to insure their devices through a device protection program. All customers have the privilege of calling in for customer support.

![alt text](https://raw.githubusercontent.com/tnorlund/CustomerAnalytics/master/Support.png "Phone Service")

Finally, each customer pays a certain amount to receive the services the telecommunications company offers. There are multiple ways for customers to pay their bills. The following graphs show the type of contract, the billing type, how the customer pays their bills, and how much the customers pay for their services.

![alt text](https://raw.githubusercontent.com/tnorlund/CustomerAnalytics/master/Billing.png "Phone Service")

### Model

Using all these features, a model can be used to predict whether a customer will leave the company or not. The model I chose to use is a Deep Neural Network. After testing a few different models, I found that the network with 1 hidden layer produced the best results.

After training the model, we can look at the how the training and validation performed.

![alt text](https://raw.githubusercontent.com/tnorlund/CustomerAnalytics/master/TrainAndValidate.png "Training and Validation")

### Results


Although the model is only able to reach 80% accuracy, the model does perform well. When applying the model to the entire dataset, we see that the model is more likely to give the false positive of predicting customer churn when in fact there is no churn.

![alt text](https://raw.githubusercontent.com/tnorlund/CustomerAnalytics/master/Confusion.png "Confusion Matrix")

