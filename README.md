
# Project Description: Visualizing The Implied Volatility Surface

In the following I will present a – not yet finished – project I am currently working on in my free time. Written Python code including comments can be found in this GitHub repository.

## Project scope and motivation
The Black-Scholes-Formula and fundamentals of financial derivatives were known to me when I came across 
this idea: Writing a script that visualizes the implied volatility surface by calculating the implied volatility for traded stock options with different specifications by using openly available data. This idea seemed like a good learning opportunity to me, because it gets someone in practice of programming, data analysis and (financial) mathematics.

## Implemented aspects
-	Finding and transforming the right data
-	Modeling the Black-Scholes-Formula in code
-	Using the Newton-algorithm for finding numerical solutions of the equation
-	Iterating over available specifications of options and saving calculated values
-	Visualizing data and calculations

## Ideas for further expansions
-	Possibility of using also put options besides call options
-	Interactive adjustment of the viewing angle in the 3D-graph
-	Using (other) color-schemes in the plots for better differentiation
-	Automating daily calculations to visualize the evolvement of implied volatility over time

## Example using Google’s (symbol "GOOG") stock option data
The script must be provided with a stock symbol, the current price of the stock, an anticipated risk-free interest rate and a lower bound for the time to maturity of options. If you execute the "ImpliedVolatilityProject.py", with possibility of importing the two other .py files containing self-defined functions, you will see the results of the following example:
For getting the data we use the Python-library “yFinance”, which fetches the data of call options on the passed symbol from the Yahoo! Finance API. When writing this, the Google stock moves around 190$ and the European Central Bank recently decreased the deposit rate to 3.00%, what will be our risk-free rate. I restricted the example to options with a minimum time to maturity of two months. This setting helps to control the extreme values plotted when time to maturity runs towards zero.
Feel free to try the code with other condigurations! :)
