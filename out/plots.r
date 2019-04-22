library(tidyverse)
library(readr)
library(plotly)
library(processx)
setwd("~/Documents/course-git-playground/Currency-Movement-Prediction/out")

# EUR ---------------------------------------------------------------------


eur_int_att <- read_csv("./USD-EUR_word2int_attention_lstm_0.csv", col_types = cols(X1 = col_skip()))
eur_vec_att <- read_csv("./USD-EUR_word2vec_attention_lstm_0.csv", col_types = cols(X1 = col_skip()))
eur_int_lstm <- read_csv("./USD-EUR_word2int_lstm_0.csv", col_types = cols(X1 = col_skip()))
eur_vec_lstm <- read_csv("./USD-EUR_word2vec_lstm_0.csv", col_types = cols(X1 = col_skip()))

eur_int_att <- eur_int_att %>% select(int_att_acc = acc, int_att_val_acc = val_acc)
eur_vec_att <- eur_vec_att %>% select(vec_att_acc = acc, vec_att_val_acc = val_acc)
eur_int_lstm <- eur_int_lstm %>% select(int_lstm_acc = acc, int_lstm_val_acc = val_acc)
eur_vec_lstm <- eur_vec_lstm %>% select(vec_lstm_acc= acc, vec_lstm_val_acc = val_acc)
eur <- bind_cols(eur_int_att, eur_int_lstm, eur_vec_att, eur_vec_lstm)

# GBP ---------------------------------------------------------------------


gbp_int_att <- read_csv("./USD-GBP_word2int_attention_lstm_0.csv", col_types = cols(X1 = col_skip()))
gbp_vec_att <- read_csv("./USD-GBP_word2vec_attention_lstm_0.csv", col_types = cols(X1 = col_skip()))
gbp_int_lstm <- read_csv("./USD-GBP_word2int_lstm_0.csv", col_types = cols(X1 = col_skip()))
gbp_vec_lstm <- read_csv("./USD-GBP_word2vec_lstm_0.csv", col_types = cols(X1 = col_skip()))


gbp_int_att <- gbp_int_att %>% select(int_att_acc = acc, int_att_val_acc = val_acc)
gbp_vec_att <- gbp_vec_att %>% select(vec_att_acc = acc, vec_att_val_acc = val_acc)
gbp_int_lstm <- gbp_int_lstm %>% select(int_lstm_acc = acc, int_lstm_val_acc = val_acc)
gbp_vec_lstm <- gbp_vec_lstm %>% select(vec_lstm_acc= acc, vec_lstm_val_acc = val_acc)
gbp <- bind_cols(gbp_int_att, gbp_int_lstm, gbp_vec_att, gbp_vec_lstm)

# CNY ---------------------------------------------------------------------


cny_int_att <- read_csv("./USD-CNY_word2int_attention_lstm_0.csv", col_types = cols(X1 = col_skip()))
cny_vec_att <- read_csv("./USD-CNY_word2vec_attention_lstm_0.csv", col_types = cols(X1 = col_skip()))
cny_int_lstm <- read_csv("./USD-CNY_word2int_lstm_0.csv", col_types = cols(X1 = col_skip()))
cny_vec_lstm <- read_csv("./USD-CNY_word2vec_lstm_0.csv", col_types = cols(X1 = col_skip()))

cny_int_att <- cny_int_att %>% select(int_att_acc = acc, int_att_val_acc = val_acc)
cny_vec_att <- cny_vec_att %>% select(vec_att_acc = acc, vec_att_val_acc = val_acc)
cny_int_lstm <- cny_int_lstm %>% select(int_lstm_acc = acc, int_lstm_val_acc = val_acc)
cny_vec_lstm <- cny_vec_lstm %>% select(vec_lstm_acc= acc, vec_lstm_val_acc = val_acc)
cny <- bind_cols(cny_int_att, cny_int_lstm, cny_vec_att, cny_vec_lstm)


# JPY ---------------------------------------------------------------------


jpy_int_att <- read_csv("./USD-JPY_word2int_attention_lstm_0.csv", col_types = cols(X1 = col_skip()))
jpy_vec_att <- read_csv("./USD-JPY_word2vec_attention_lstm_0.csv", col_types = cols(X1 = col_skip()))
jpy_int_lstm <- read_csv("./USD-JPY_word2int_lstm_0.csv", col_types = cols(X1 = col_skip()))
jpy_vec_lstm <- read_csv("./USD-JPY_word2vec_lstm_0.csv", col_types = cols(X1 = col_skip()))

jpy_int_att <- jpy_int_att %>% select(int_att_acc = acc, int_att_val_acc = val_acc)
jpy_vec_att <- jpy_vec_att %>% select(vec_att_acc = acc, vec_att_val_acc = val_acc)
jpy_int_lstm <- jpy_int_lstm %>% select(int_lstm_acc = acc, int_lstm_val_acc = val_acc)
jpy_vec_lstm <- jpy_vec_lstm %>% select(vec_lstm_acc= acc, vec_lstm_val_acc = val_acc)
jpy <- bind_cols(jpy_int_att, jpy_int_lstm, jpy_vec_att, jpy_vec_lstm)


# BTC ---------------------------------------------------------------------


btc_int_att <- read_csv("./USD-BTC_word2int_attention_lstm_0.csv", col_types = cols(X1 = col_skip()))
btc_vec_att <- read_csv("./USD-BTC_word2vec_attention_lstm_0.csv", col_types = cols(X1 = col_skip()))
btc_int_lstm <- read_csv("./USD-BTC_word2int_lstm_0.csv", col_types = cols(X1 = col_skip()))
btc_vec_lstm <- read_csv("./USD-BTC_word2vec_lstm_0.csv", col_types = cols(X1 = col_skip()))

btc_int_att <- btc_int_att %>% select(int_att_acc = acc, int_att_val_acc = val_acc)
btc_vec_att <- btc_vec_att %>% select(vec_att_acc = acc, vec_att_val_acc = val_acc)
btc_int_lstm <- btc_int_lstm %>% select(int_lstm_acc = acc, int_lstm_val_acc = val_acc)
btc_vec_lstm <- btc_vec_lstm %>% select(vec_lstm_acc= acc, vec_lstm_val_acc = val_acc)
btc <- bind_cols(btc_int_att, btc_int_lstm, btc_vec_att, btc_vec_lstm)

# Plot --------------------------------------------------------------------


x <- c(1:100)
eur_plot <- plot_ly(data = eur, type="scatter") %>% 
  add_trace(y = ~int_att_acc, name = "int Attention-LSTM", mode="lines", line = list(color="red"))%>% 
  add_trace(y = ~vec_att_acc, name = "vec Attention-LSTM", mode="lines", line = list(color="blue")) %>% 
  add_trace(y = ~int_lstm_acc, name = "int LSTM", mode="lines", line = list(color="orange")) %>% 
  add_trace(y = ~vec_lstm_acc, name = "vec LSTM", mode="lines", line=list(color="purple")) %>% 
  add_trace(y = ~int_att_val_acc, name = "int Attention-LSTM (Val)", mode="markers", marker = list(color="red"))%>% 
  add_trace(y = ~vec_att_val_acc, name = "vec Attention-LSTM (Val)", mode="markers", marker = list(color="blue")) %>% 
  add_trace(y = ~int_lstm_val_acc, name = "int LSTM (Val)", mode="markers", marker = list(color="orange")) %>% 
  add_trace(y = ~vec_lstm_val_acc, name = "vec LSTM (Val)", mode="markers", marker=list(color="purple")) %>% 
  layout(title = "USD-EUR",
         xaxis = list(title = "Epoch"),
         yaxis = list(title = "Accuracy"),
         legend = list(orientation = "v", x = 0.7, y = 0))


gbp_plot <- plot_ly(data = gbp, type="scatter") %>% 
  add_trace(y = ~int_att_acc, name = "int Attention-LSTM", mode="lines", line = list(color="red"))%>% 
  add_trace(y = ~vec_att_acc, name = "vec Attention-LSTM", mode="lines", line = list(color="blue")) %>% 
  add_trace(y = ~int_lstm_acc, name = "int LSTM", mode="lines", line = list(color="orange")) %>% 
  add_trace(y = ~vec_lstm_acc, name = "vec LSTM", mode="lines", line=list(color="purple")) %>% 
  add_trace(y = ~int_att_val_acc, name = "int Attention-LSTM (Val)", mode="markers", marker = list(color="red"))%>% 
  add_trace(y = ~vec_att_val_acc, name = "vec Attention-LSTM (Val)", mode="markers", marker = list(color="blue")) %>% 
  add_trace(y = ~int_lstm_val_acc, name = "int LSTM (Val)", mode="markers", marker = list(color="orange")) %>% 
  add_trace(y = ~vec_lstm_val_acc, name = "vec LSTM (Val)", mode="markers", marker=list(color="purple")) %>% 
  layout(title = "USD-GBP",
         xaxis = list(title = "Epoch"),
         yaxis = list(title = "Accuracy"),
         legend = list(orientation = "v", x = 0.7, y = 0))
  
cny_plot <- plot_ly(data = cny, type="scatter") %>% 
  add_trace(y = ~int_att_acc, name = "int Attention-LSTM", mode="lines", line = list(color="red"))%>% 
  add_trace(y = ~vec_att_acc, name = "vec Attention-LSTM", mode="lines", line = list(color="blue")) %>% 
  add_trace(y = ~int_lstm_acc, name = "int LSTM", mode="lines", line = list(color="orange")) %>% 
  add_trace(y = ~vec_lstm_acc, name = "vec LSTM", mode="lines", line=list(color="purple")) %>% 
  add_trace(y = ~int_att_val_acc, name = "int Attention-LSTM (Val)", mode="markers", marker = list(color="red"))%>% 
  add_trace(y = ~vec_att_val_acc, name = "vec Attention-LSTM (Val)", mode="markers", marker = list(color="blue")) %>% 
  add_trace(y = ~int_lstm_val_acc, name = "int LSTM (Val)", mode="markers", marker = list(color="orange")) %>% 
  add_trace(y = ~vec_lstm_val_acc, name = "vec LSTM (Val)", mode="markers", marker=list(color="purple")) %>% 
  layout(title = "USD-CNY",
         xaxis = list(title = "Epoch"),
         yaxis = list(title = "Accuracy"),
         legend = list(orientation = "v", x = 0.7, y = 0))

jpy_plot <- plot_ly(data = jpy, type="scatter") %>% 
  add_trace(y = ~int_att_acc, name = "int Attention-LSTM", mode="lines", line = list(color="red"))%>% 
  add_trace(y = ~vec_att_acc, name = "vec Attention-LSTM", mode="lines", line = list(color="blue")) %>% 
  add_trace(y = ~int_lstm_acc, name = "int LSTM", mode="lines", line = list(color="orange")) %>% 
  add_trace(y = ~vec_lstm_acc, name = "vec LSTM", mode="lines", line=list(color="purple")) %>% 
  add_trace(y = ~int_att_val_acc, name = "int Attention-LSTM (Val)", mode="markers", marker = list(color="red"))%>% 
  add_trace(y = ~vec_att_val_acc, name = "vec Attention-LSTM (Val)", mode="markers", marker = list(color="blue")) %>% 
  add_trace(y = ~int_lstm_val_acc, name = "int LSTM (Val)", mode="markers", marker = list(color="orange")) %>% 
  add_trace(y = ~vec_lstm_val_acc, name = "vec LSTM (Val)", mode="markers", marker=list(color="purple")) %>% 
  layout(title = "USD-JPY",
         xaxis = list(title = "Epoch"),
         yaxis = list(title = "Accuracy"),
         legend = list(orientation = "v", x = 0.7, y = 0))

btc_plot <- plot_ly(data = btc, type="scatter") %>% 
  add_trace(y = ~int_att_acc, name = "int Attention-LSTM", mode="lines", line = list(color="red"))%>% 
  add_trace(y = ~vec_att_acc, name = "vec Attention-LSTM", mode="lines", line = list(color="blue")) %>% 
  add_trace(y = ~int_lstm_acc, name = "int LSTM", mode="lines", line = list(color="orange")) %>% 
  add_trace(y = ~vec_lstm_acc, name = "vec LSTM", mode="lines", line=list(color="purple")) %>% 
  add_trace(y = ~int_att_val_acc, name = "int Attention-LSTM (Val)", mode="markers", marker = list(color="red"))%>% 
  add_trace(y = ~vec_att_val_acc, name = "vec Attention-LSTM (Val)", mode="markers", marker = list(color="blue")) %>% 
  add_trace(y = ~int_lstm_val_acc, name = "int LSTM (Val)", mode="markers", marker = list(color="orange")) %>% 
  add_trace(y = ~vec_lstm_val_acc, name = "vec LSTM (Val)", mode="markers", marker=list(color="purple")) %>% 
  layout(title = "USD-BTC",
         xaxis = list(title = "Epoch"),
         yaxis = list(title = "Accuracy"),
         legend = list(orientation = "v", x = 0.7, y = 0))


# Export ------------------------------------------------------------------

# This section requires the `orca` command line utility from plot.ly 
# Plot PNGs can still be exported by RStudio

orca(eur_plot, "./eur_plot.png", width = 640, height = 480)
orca(cny_plot, "./cny_plot.png", width = 640, height = 480)
orca(jpy_plot, "./jpy_plot.png", width = 640, height = 480)
orca(btc_plot, "./btc_plot.png", width = 640, height = 480)
orca(gbp_plot, "./gbp_plot.png", width = 640, height = 480)

