import streamlit as st

# Custom imports
from src.app import MultiPage
from src.app import explore_infoscc_gan, explore_biggan, explore_cvae, compare_models

# Create an instance of the app
app = MultiPage()

# Title of the main page
st.title('Galaxy Zoo generation')

# Add all your applications (pages) here
app.add_page('Compare models', compare_models.app)
app.add_page('Explore BigGAN', explore_biggan.app)
app.add_page('Explore cVAE', explore_cvae.app)
app.add_page('Explore InfoSCC-GAN', explore_infoscc_gan.app)

# The main app
app.run()
