"""
movie_data_helper.py
Original author: Felix Sun (6.008 TA, Fall 2015)
Modified by:
- Danielle Pace (6.008 TA, Fall 2016),
- George H. Chen (6.008/6.008.1x instructor, Fall 2016)

***Do not modify this file.***

This file has a number of helper files for the movie rating problem, to
interact with the data files and return relevant information.
"""

import numpy as np
from os.path import isfile
from sys import exit


def get_movie_id_list():
    """
    This function returns a 1D NumPy array of all movie ID's based on data in
    './data/'.

    Output
    ------
    1D NumPy array of all the movie ID's
    """

    return np.loadtxt("data/movieNames.dat",
                      dtype='int32',
                      delimiter='\t',
                      usecols=(0,))


def get_movie_name(movie_id):
    """
    Gets a movie name given a movie ID.

    Input
    -----
    - movie_id: integer from 0 to <number of movies - 1> specifying which movie

    Output
    ------
    - movie_name: string containing the movie name
    """

    # -------------------------------------------------------------------------
    # ERROR CHECK
    #
    filename_to_check = "data/ratingsMovie%d.dat" % movie_id
    if not(isfile(filename_to_check)):
        exit('Movie ID %d does not exist' % movie_id)
    #
    # END OF ERROR CHECK
    # -------------------------------------------------------------------------

    filename = "data/movieNames.dat"
    movies = np.loadtxt(filename,
                        dtype={'names': ('movieid', 'moviename'),
                               'formats': ('int32', 'S100')},
                        delimiter='\t')
    movie_indices = [i for i in range(len(movies))
                     if (movies[i]['movieid'] == movie_id)]
    movie_name = movies[movie_indices]['moviename'][0].lstrip()
    return movie_name


def get_ratings(movie_id):
    """
    Gets all the ratings for a given movie.

    Input
    -----
    - movie_id: integer from 0 to <number of movies - 1> specifying which movie

    Output
    ------
    - ratings: 1D array consisting of the ratings for the given movie
    """

    filename = "data/ratingsMovie%d.dat" % movie_id

    # -------------------------------------------------------------------------
    # ERROR CHECK
    #
    if not(isfile(filename)):
        exit('Movie ID %d does not exist' % movie_id)
    #
    # END OF ERROR CHECK
    # -------------------------------------------------------------------------

    data = np.loadtxt(filename,
                      dtype={'names': ('userid', 'rating'),
                             'formats': ('int32', 'int32')},
                      delimiter='\t')
    ratings = data['rating']
    return ratings
