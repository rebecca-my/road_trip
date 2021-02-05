## Data
This is a simple dataset of North American (though mostly U.S.) major roads.

`city_gps.txt` contains one line per city, with three fields per line, 
delimited by spaces. The first field is the city, followed by the latitude,
followed by the longitude.

`segments.txt` has one line per road segment connecting two cities.
The space delimited fields are:

- first city
- second city
- length (in miles)
- speed limit (in miles per hour)
- name of highway

All roads in `segments.txt` are bidirectional, i.e. none are one-way roads, so
that it's possible to travel from the first city to the second city at the
same distance at speed as from the second city to the first city.

## Program
Run 
```
python3 ./route.py [start-city] [end-city] [cost-function]

```
## cost function is one of:
– `segments` tries to find a route with the fewest number of road segments (i.e. edges of the graph).
– `distance` tries to find a route with the shortest total distance.
– `time` finds the fastest route, for a car that always travelsfive miles per hour above the speed limit.
