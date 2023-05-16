# Surgical Schedule Optimization

This project optimizes the scheduling of surgeries in a hospital to minimize staff dissatisfaction while meeting resource requirements.

## Data

The data used in this project is in the form of a CSV file with the following columns:

- `Case Type`: The type of surgery being performed
- `Theatre`: The operating room where the surgery will take place
- `Surgery Start Date`: The date the surgery is scheduled to begin
- `Surgery End Date`: The date the surgery is scheduled to end
- `Anaesthetic Start`: The time anesthesia is scheduled to begin
- `Surgery Start`: The time the surgery is scheduled to begin
- `Surgery Finish`: The time the surgery is scheduled to end
- `Anaesthetic Finish`: The time anesthesia is scheduled to end
- `Left Theatre`: The time the patient is scheduled to leave the operating room

## Usage

To run the optimization algorithm, execute the `main()` function in `optimize.py`. The result will be the best solution found by the algorithm.

## Dependencies

- pandas
- numpy
- ortools
- deap
- simanneal

Install dependencies with `pip install -r requirements.txt`.
