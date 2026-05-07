from src.api import app  # imports the app object from api.py.

print(app.url_map)

if __name__ == "__main__":
    app.run(debug=True)  # starts the Flask development server.
