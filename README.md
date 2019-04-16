# Currency-Movement-Prediction

### Summary

This project is a LSTM-based Recurrent Neural Network that predicts currency exchange market variations based on financial/political news articles. The articles will be preprocessed (format cleanup, entity extraction, etc) and then be fed into the neural network. The expected output of this neural network is the predicted variations of the major currencies based on the news article(s) given. 

### Preprocessing

* Features 
Currency Data: Date, GBP, Day-Before, Day-After, Week-Before, Week-After
News Data: Title, Date, Content, Year, Month, Day


`Google hit with €1.5bn fine from EU over advertising` -> `EU (EU) neg Google (US)` -> `USD goes down, EUR goes up`
`Trump declares national emergency to fund the wall` -> `Trump (US) neg` -> `USD goes down`

### Questions:

1. We need to find a way to map entities mentioned in news (company, people, etc) to countries they belonged to. For example, in the news "Google hit with €1.5bn fine from EU over advertising", we need to find a way to map Google to US before performing sentiment analysis. 
2. How will entities be represented in the input (will word2vec still work in this case?)
3. How to evaluate/ Baselines to use. There are research on news sentiment vs. stock/cryptocurrency movements but nothing for news sentiment vs. currency movements. What should we use for baseline besides random?
4. Common parameters to tune for LSTM?


openexchangerates