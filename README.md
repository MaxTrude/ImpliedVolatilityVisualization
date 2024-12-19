
#Project Description: Visualizing The Implied Volatility Surface

In the following I will present a – not yet finished – project I am currently working on in my free time. Written Python code including comments can be found in this GitHub repository.

Project scope and motivation
The Black-Scholes-Formula and fundamentals of financial derivatives were known to me when I came across 
this idea: Writing a script that visualizes the implied volatility surface by calculating the implied volatility for traded stock options with different specifications by using openly available data. This idea seemed like a good learning opportunity to me, because it gets someone in practice of programming, data analysis and (financial) mathematics.

Implemented aspects
-	Finding and transforming the right data
-	Modeling the Black-Scholes-Formula in code
-	Using the Newton-algorithm for finding numerical solutions of the equation
-	Iterating over available specifications of options and saving calculated values
-	Visualizing data and calculations

Ideas for further expansions
-	Possibility of using also put options besides call options
-	Interpolation of the data points to a surface in the 3D-graph
-	Interactive adjustment of the viewing angle in the 3D-graph
-	Using color-schemes in the plots for better differentiation
-	Automating daily calculations to visualize the evolvement of implied volatility over time

Example using Google’s stock option data
The script must be provided with a stock symbol, the current price of the stock, an anticipated risk-free interest rate and a lower bound for the time to maturity of options.
For getting the data we use the Python-library “yFinance”, which fetches the data of call options on the passed symbol from the Yahoo! Finance API. When writing this, the Google stock moves around 185$ and the European Central Bank recently decreased the deposit rate to 3.00%, what will be our risk-free rate. We won’t restrict the example to options of a minimum time to maturity. However, this setting can help to control the points plotted when time to maturity runs towards zero. We receive the following plots after running the script:
 
 	Aachen, 19.12.2024

