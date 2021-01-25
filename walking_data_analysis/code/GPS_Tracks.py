import sys
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from math import cos, asin, sqrt, pi

# modified from e3
class GPSTracker:
    file_name = "N/A"
    filtered_distance = 0

    def __init__(self, file_name):
        self.file_name = file_name

    def get_data(self, file):
        # df = pd.read_csv(file)
        df = file
        gps_data = {'lat': df['Latitude'], 'lon' : df['Longitude']}
        data = pd.DataFrame(gps_data, columns=['lat', 'lon'])
        return data


    def distance_pair(self, lat1, lon1, lat2, lon2):
        p = pi / 180
        a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
        return 12742 * asin(sqrt(a))


    def distance(self, df):
        distance_pairs = np.vectorize(self.distance_pair)
        df['latNext'] = df['lat'].shift(periods=-1)
        df['lonNext'] = df['lon'].shift(periods=-1)
        df['distance'] = distance_pairs(df['lat'], df['lon'], df['latNext'], df['lonNext'])
        del df['latNext']
        del df['lonNext']
        total = df['distance'].sum() * 1000
        del df['distance']
        return total


    def smooth(self, df):
        initial_state = df.iloc[0]
        observation_covariance = np.diag([0.001, 0.001]) ** 2
        transition_covariance = np.diag([0.0005, 0.0005]) ** 2
        transition = [[1, 0], [0, 1]]

        kf = KalmanFilter(
            initial_state_mean=initial_state,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition)
        kalman_smoothed, _ = kf.smooth(df)
        df = pd.DataFrame(kalman_smoothed, columns=['lat', 'lon'])
        return df


    def output_gpx(self, points, output_filename):
        """
        Output a GPX file with latitude and longitude from the points DataFrame.
        """
        from xml.dom.minidom import getDOMImplementation
        def append_trkpt(pt, trkseg, doc):
            trkpt = doc.createElement('trkpt')
            trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
            trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
            trkseg.appendChild(trkpt)

        doc = getDOMImplementation().createDocument(None, 'gpx', None)
        trk = doc.createElement('trk')
        doc.documentElement.appendChild(trk)
        trkseg = doc.createElement('trkseg')
        trk.appendChild(trkseg)

        points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)

        with open(output_filename, 'w') as fh:
            doc.writexml(fh, indent=' ')

    def get_distance(self):
        points = self.get_data(self.file_name)
        print('Unfiltered distance: %0.2f' % (self.distance(points),))
        smoothed_points = self.smooth(points)
        self.filtered_distance = self.distance(smoothed_points)
        print('Filtered distance: %0.2f' % (self.filtered_distance,))
        return self.filtered_distance

    def out_gpxfile(self):
        points = self.get_data(self.file_name)
        print('Unfiltered distance: %0.2f' % (self.distance(points),))

        smoothed_points = self.smooth(points)
        print('Filtered distance: %0.2f' % (self.distance(smoothed_points),))
        self.output_gpx(smoothed_points, 'out.gpx')

