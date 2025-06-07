# Weather Forecasting System - Clean Architecture Implementation

This project implements a weather forecasting system for Rio Guaíba using Clean Architecture principles. The system is designed to predict precipitation levels 24 hours in advance based on historical weather data.

## Architecture

The project follows Clean Architecture with the following layers:

- **Domain Layer**: Core business entities and interfaces (WeatherData, Forecast, repositories)
- **Application Layer**: Use cases that orchestrate business logic (GenerateForecastUseCase)
- **Infrastructure Layer**: Concrete implementations of repositories and external services
- **Presentation Layer**: API endpoints and data transformation (FastAPI routes)

## Testing the Implementation

To verify the implementation and test the system, follow these steps:

### Prerequisites

- Python 3.9+
- Required packages: `fastapi`, `uvicorn`, `pydantic-settings` (for full functionality)

Install the required packages:

```bash
pip install fastapi uvicorn pydantic-settings
```

### Running the Tests

1. **Run the Unit Tests**:

```bash
python run_tests.py
```

This will run all the unit tests and verify that individual components are working correctly.

2. **Test Repository Implementations**:

```bash
python test_repo.py
```

This script tests the various repository implementations to ensure they can correctly store and retrieve data.

3. **Test End-to-End Functionality**:

```bash
python simple_check.py
```

This script demonstrates the entire system working together by manually wiring dependencies and generating a forecast.

### Test Results

For detailed test results and analysis of the Clean Architecture implementation, see [TEST_RESULTS.md](TEST_RESULTS.md).

## Project Structure

```
app/
  ├── features/
  │   └── forecast/
  │       ├── domain/         # Entities, repositories interfaces
  │       ├── application/    # Use cases
  │       ├── infrastructure/ # Repository implementations
  │       └── presentation/   # API endpoints
  └── main.py                 # Application entry point
data/
  └── processed/              # Data files
models/                       # Model files
tests/                        # Unit tests
```

## Running the Application

To run the full application (requires FastAPI):

```bash
uvicorn app.main:app --reload
```

Then visit `http://127.0.0.1:8000/docs` to see the API documentation.

## Notes

- The system is designed to work with TensorFlow models, but can run in compatibility mode without TensorFlow installed.
- For testing purposes, mock implementations are provided that don't require TensorFlow.
- The implementation demonstrates how Clean Architecture allows for separation of concerns and testability.
