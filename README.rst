==============================================================================
Gaussian Process based spatio-temporal modeling of pedestrians
==============================================================================
Prerequisites
=============
 - Python 3.8

On macOS it might be necessary to install SSL Certificates for Python, by running this command in terminal ::

    open /Applications/Python\ 3.8/Install\ Certificates.command

Installation
============
Install via ::

    python setup.py install

For the Mongo Interface to work, the credentials must be placed in the ``src/cmr_people_gp/data_io folder``.
Fill the file with your credentials with the following command.
Note that USERNAME/PASSWORD must be replaced by your specific credentials. You can find your credentials in the ``passwords.kdbx`` file in ``cmr_utils``.
Use your user credentials from the folder user_passwords (not vpn!)
Special characters (e.g. ``!``) in USERNAME or PASSWORD must be escaped (use e.g. ``\!`` instead of ``!``) ::


    echo "user: \"USERNAME\"\npassword: \"PASSWORD\"" >> src/cmr_people_gp/data_io/mongo.user.yaml

The file ``src/cmr_people_gp/data_io/mongo.user.yaml`` should then have the following content ::

    user: USERNAME
    password: PASSWORD

Run tests::

    python setup.py test
