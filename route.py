import pandas as pd
import numpy as np
import os
from math import sin, cos, sqrt, atan2, radians
import sys

class Astar():
    """docstring for Astar"""
    def __init__(self, environment):
        self.environment = environment

    def _pop_best_fringe(self):
        best = None
        best_score = 1e999
        for name, (parent, g, h) in self.fringe.items():
            if g+h < best_score:
                best = (name, parent, g, h)
                best_score = g+h
        del self.fringe[best[0]]
        self.history.append(best)
        return best

    def _report(self, city):
        parent, _, _ = self.visited[city]
        route = [city]
        while parent is not None:
            route.append(parent)
            parent, _, _ = self.visited[parent]
        return self.environment.generate_report(list(reversed(route)))

    def solve(self, initial_city, final_city, metric):
        # Initialize
        calc_h = self.environment.get_calc_h(metric, final_city)
        calc_g = self.environment.get_calc_g(metric)

        self.history = []
        self.visited = {} # keys are nodes, values are best (parent, g, h)
        self.fringe = {} # keys are nodes, values are proposed (parent, g, h)
        self.fringe[initial_city] = (None, 0, calc_h(initial_city))

        # Run
        while len(self.fringe) > 0:
            # move best fringe to visited & check if done
            city, parent, city_g, city_h = self._pop_best_fringe()
            self.visited[city] = (parent, city_g, city_h)
            if city == final_city:
                return self._report(city)

            # expand children, skipping already optimal ones (e.g. in visited)
            for succ in self.environment.successors(city):
                if succ not in self.visited.keys():

                    # add child to fringe, or update if better
                    new_g = calc_g(succ, city, city_g)
                    if succ not in self.fringe.keys():
                        self.fringe[succ] = (city, new_g, calc_h(succ))
                    else:
                        old_parent, old_g, h = self.fringe[succ]
                        if new_g < old_g:
                            self.fringe[succ] = (city, new_g, h)  

        return False


class LongNetwork():
    def __init__(self, n, k, r=float('inf')):
        self.neighs = {}
        self.r = r
        for k, v in enumerate(np.random.choice(n,(n,k))):
            v = set(v[np.where(abs(v-k)<r)])
            self.neighs[k] = v
                
    def get_calc_h(self, metric, target):
        def zeros(node):
            return abs(node-target)/self.r
        return zeros

    def get_calc_g(self, metric):
        def ones(node, parent, parent_g):
            return 1 + parent_g
        return ones

    def successors(self, node):
        return self.neighs[node]

    
class RoadNetwork():
    def __init__(self, path = ''):
        road_file = os.path.join(path, 'road-segments.txt')
        roads = pd.read_csv(road_file, sep = ' ', header = None)
        roads.columns = ['start', 'end', 'distance', 'speed', 'name']
        roads_dummy = roads.copy()
        roads_dummy.columns = ['end', 'start', 'distance', 'speed', 'name']
        self.roads = pd.concat([roads_dummy.sort_index(axis=1), roads.sort_index(axis=1)])
        self.roads['time'] = self.roads['distance']/(self.roads['speed']+5)*1 # hours
        self.roads['risk'] = self.roads['distance']*self.roads['speed']

        self.min_risk = self.roads['risk'].min()
        self.max_segment_len = self.roads['distance'].max()
        self.max_speed = self.roads['speed'].max()
        self.min_speed = self.roads['speed'].min()

        city_file = os.path.join(path, 'city-gps.txt')
        cities = pd.read_csv(city_file, sep=' ', header = None)
        cities.columns = ['names', 'lat', 'long']
        self.cities = cities.set_index('names')

    def generate_report(self, route):
        #[total-segments] [total-miles] [total-hours] [total-expected-accidents] [start-city] [city-1] [city-2] ... [end-city]
        pairs = [(start, end) for start, end in zip(route[:-1],route[1:])]
        self.roads['special_index'] = list(zip(self.roads['start'], self.roads['end']))
        df_report = self.roads.set_index('special_index').loc[pairs]
        df_report = df_report.reset_index()[['name','distance','time','risk','start','end']]
        df_report['risk']*=0.000001
        #print('\t'.join(df_report.columns))
        # for row in df_report.iterrows():
        #     print('\t'.join(str(item) for item in row))
        print(df_report.to_string(index=False, index_names=False))
        total_segments = len(df_report)
        total_miles = df_report['distance'].sum()
        total_time = round(df_report['time'].sum(), 3)
        total_risk = round(df_report['risk'].sum(), 3)
        print(total_segments, total_time, total_miles, total_risk, *route)

    def get_calc_h(self, metric, final):
        nearest_city = self.roads.groupby('end')['distance'].min()

        def distance(city):
            R = 3958.8
            if city in self.cities.index and final in self.cities.index:
                lat1, lon1 = self.cities.loc[city]
                lat2, lon2 = self.cities.loc[final]
                try:
                    dlon = radians(lon2) - radians(lon1)
                    dlat = radians(lat2) - radians(lon2)
                except TypeError:
                    print(city,final)
                    raise TypeError
                a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
                c = 2 * atan2(sqrt(a), sqrt(1 - a))
                distance = R * c
            else:
                distance = nearest_city[city]
            return np.floor(distance*0.95)
            #return 0
            #return None ## this is going ot be the calculated dist between city and final based on lat and lon

        def heuristic_dist(city):
            d = distance(city)
            if d is None:
               return 0
            else:
                return d

        def heuristic_time(city):
            d = distance(city)
            if d is None:
                return 0 ## maybe there is something better
            else:
                return d / (self.max_speed+5)

        def heuristic_risk(city):
            d = distance(city)
            if d is None:
                return self.min_risk ## maybe there is something better
            else:
                return d * self.min_speed

        def heuristic_segments(city):
            d = distance(city)
            if d is None:
                if city == final:
                    return 0
                else:
                    return 1 ## maybe there is something better
            else:
                return d / self.max_segment_len

        if metric =='distance':
            return heuristic_dist

        if metric =='speed':
            return heuristic_time

        if metric =='cycling':
            return heuristic_risk

        if metric =='segments':
            return heuristic_segments

    def get_calc_g(self, metric):

        def calc_g_distance(city_a, city_b, parent_g):
            return self.roads[np.logical_and(self.roads['start'] == city_a,
                                             self.roads['end']   == city_b)]['distance'].values[0] + parent_g

        def calc_g_segments(city_a, city_b, parent_g):
            return parent_g+1

        def calc_g_time(city_a, city_b, parent_g):
            return self.roads[np.logical_and(self.roads['start'] == city_a,
                                             self.roads['end']   == city_b)]['time'].values[0] + parent_g

        def calc_g_risk(city_a, city_b, parent_g):
            return self.roads[np.logical_and(self.roads['start'] == city_a,
                                             self.roads['end']   == city_b)]['risk'].values[0] + parent_g

        if metric =='distance':
            return calc_g_distance

        if metric =='speed':
            return calc_g_time

        if metric =='cycling':
            return calc_g_risk

        if metric =='segments':
            return calc_g_segments

    def successors(self, city):
        return self.roads[self.roads['start']==city]['end'].tolist()

if __name__ == "__main__":
    if(len(sys.argv) != 4):
        raise(Exception("Error: expected 3 inputs"))

    network = RoadNetwork()
    ss = Astar(network)
    ss.solve(sys.argv[1], sys.argv[2], sys.argv[3])