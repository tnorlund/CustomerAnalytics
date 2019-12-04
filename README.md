
# Customer Analytics

## Campaign Effectiveness

Advertising campaigns have the possibility of being extremely effective or a waste of money.

Companies can compare how customers respond to specific advertising campaigns by using different campaigns for different customers or store locations. The following data is an example of how a business can use different advertising campaigns for different markets.

### Data

Each advertising campaign was tried at different locations. The following graph shows how the different locations produce different results (sales).

![alt text](https://raw.githubusercontent.com/tnorlund/CustomerAnalytics/master/SalesPerLocation.png "Sales Per Location")

Here we see that some stores perform better than others. Continuing the exploratory analysis, the results of different sized stores can be compared.

![alt text](https://raw.githubusercontent.com/tnorlund/CustomerAnalytics/master/SalesMarketSize.png "Sales Market Size")

We can see that the larger sized stores perform the best. Not only does the size of the store determine the sales, but it also determines how much money is required to keep the store running. If the cost of maintaining the stores were given, we would be able to give a better estimate surrounding what stores to keep and which to drop. 

We can continue the data analysis by looking at how the age of the store effects sales.

![alt text](https://raw.githubusercontent.com/tnorlund/CustomerAnalytics/master/SalesStoreAge.png "Sales Age of Store")

Here we see that the best performing stores are between 10 and 15 years old. 


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

