<h1 align="center">Board Game Ratings D3 Choropleth Visualization Map</h1>

This project focuses on building an interactive choropleth map using D3 that shows the average board game ratings for a wide range of board games based on each country. The data is provided by the Board Game Geek database. Here is the description of the project files.

- lib: contains D3.js files that will be referenced for this project

- Board_Game_Ratings: Contains the data files and final html file for the visualization map.
  - ratings-by-country.csv: Contains the raw data of all board game average rating by country and board game. Also details number of users who rated the board game
  - world_countries.json: A JSON file that contains the necessary coordinates of the countries represented in a world map. This file is used by the D3 functions and methods to create the map
  - choroopleth.html: Main html file that builds the choropleth map using D3 libraries, parses the board game data, and integrates both of them to build the final interactive map. Includes a color scale to represent different quantities as well as a dropdown menu to let users choose different board game options

Steps on how to interact with this visualization:

- Create a local http.server to run the files from the local directory. All paths are relative to working directory

- Open choropleth.html in browser of choice
  - From dropdown menu, you can choose to see the average rating for any board game
  - Can also view more details by hovering over each country with cursor
