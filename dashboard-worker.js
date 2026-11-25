const fields = [
  "Age", "Gender", "Emirate", "Diagnosis_Date", "Cancer_Type",
  "Cancer_Stage", "Outcome", "Smoking_Status", "Nationality"
];
const filterFields = ["Cancer_Type", "Emirate", "Gender", "Outcome"];
const groupFields = ["Cancer_Type", "Cancer_Stage", "Emirate", "Smoking_Status", "Nationality"];
const allCategoryFields = [...new Set([...filterFields, ...groupFields])];
let records = [];
let options = {};
let totalRecords = 0;

function* parseCsv(text) {
  let row = [];
  let value = "";
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (char === '"') {
      if (quoted && text[i + 1] === '"') {
        value += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      row.push(value);
      value = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && text[i + 1] === "\n") i += 1;
      row.push(value);
      value = "";
      if (row.some(cell => cell !== "")) yield row;
      row = [];
    } else {
      value += char;
    }
  }
  if (value !== "" || row.length) {
    row.push(value);
    if (row.some(cell => cell !== "")) yield row;
  }
}

function indexOfColumn(header, name) {
  const index = header.indexOf(name);
  if (index < 0) throw new Error(`The dataset is missing the ${name} column.`);
  return index;
}

async function loadDataset() {
  const dataUrl = new URL("data/processed/cleaned_dataset.csv", self.location.href);
  const response = await fetch(dataUrl);
  if (!response.ok) throw new Error(`Could not load dashboard data (${response.status}).`);
  const csv = await response.text();
  const rows = parseCsv(csv);
  const header = rows.next().value;
  if (!header) throw new Error("The dashboard dataset is empty.");
  const indices = Object.fromEntries(fields.map(name => [name, indexOfColumn(header, name)]));
  const found = Object.fromEntries([...filterFields, ...groupFields].map(name => [name, new Set()]));
  let minAge = Infinity;
  let maxAge = -Infinity;
  let minYear = Infinity;
  let maxYear = -Infinity;

  for (const row of rows) {
    const age = row[indices.Age] === "" ? NaN : Number(row[indices.Age]);
    const date = row[indices.Diagnosis_Date] || "";
    const year = /^\d{4}/.test(date) ? Number(date.slice(0, 4)) : NaN;
    const categoryValues = {};
    for (const name of allCategoryFields) {
      const value = row[indices[name]] || "";
      categoryValues[name] = value;
      if (value) found[name].add(value);
    }
    records.push([
      Number.isFinite(age) ? age : null,
      Number.isFinite(year) ? year : null,
      categoryValues.Gender,
      categoryValues.Emirate,
      categoryValues.Cancer_Type,
      categoryValues.Outcome,
      categoryValues.Cancer_Stage,
      categoryValues.Smoking_Status,
      categoryValues.Nationality
    ]);
    if (Number.isFinite(age)) {
      minAge = Math.min(minAge, age);
      maxAge = Math.max(maxAge, age);
    }
    if (Number.isFinite(year)) {
      minYear = Math.min(minYear, year);
      maxYear = Math.max(maxYear, year);
    }
  }

  totalRecords = records.length;
  options = Object.fromEntries([...filterFields, ...groupFields].map(name => [name, [...found[name]].sort((a, b) => a.localeCompare(b))]));
  self.postMessage({
    type: "ready",
    total: totalRecords,
    options,
    ranges: {
      age: [minAge, maxAge],
      year: [minYear, maxYear]
    }
  });
}

function summarize(filters) {
  const selected = Object.fromEntries(filterFields.map(name => [name, new Set(filters[name])]));
  let matchingCount = 0;
  const outcomeCounts = new Map();
  const stageCounts = new Map();
  const yearCounts = new Map();
  const breakdownCounts = new Map();
  const cancers = new Set();
  const emirates = new Set();
  const ages = [];
  const breakdownField = groupFields.includes(filters.breakdown) ? filters.breakdown : "Cancer_Type";
  const columnFor = { Gender: 2, Emirate: 3, Cancer_Type: 4, Outcome: 5, Cancer_Stage: 6, Smoking_Status: 7, Nationality: 8 };

  for (const record of records) {
    const [age, year, gender, emirate, cancer, outcome, stage, smoking, nationality] = record;
    if (age === null || year === null || age < filters.age[0] || age > filters.age[1] || year < filters.year[0] || year > filters.year[1]) continue;
    if (!selected.Cancer_Type.has(cancer) || !selected.Emirate.has(emirate) || !selected.Gender.has(gender) || !selected.Outcome.has(outcome)) continue;

    matchingCount += 1;
    ages.push(age);
    cancers.add(cancer);
    emirates.add(emirate);
    outcomeCounts.set(outcome, (outcomeCounts.get(outcome) || 0) + 1);
    yearCounts.set(year, (yearCounts.get(year) || 0) + 1);

    if (!stageCounts.has(stage)) stageCounts.set(stage, new Map());
    const outcomesForStage = stageCounts.get(stage);
    outcomesForStage.set(outcome, (outcomesForStage.get(outcome) || 0) + 1);

    const group = record[columnFor[breakdownField]] || "Unknown";
    if (!breakdownCounts.has(group)) breakdownCounts.set(group, new Map());
    const outcomesForGroup = breakdownCounts.get(group);
    outcomesForGroup.set(outcome, (outcomesForGroup.get(outcome) || 0) + 1);
  }

  ages.sort((a, b) => a - b);
  const middle = Math.floor(ages.length / 2);
  const median = ages.length ? (ages.length % 2 ? ages[middle] : (ages[middle - 1] + ages[middle]) / 2) : null;
  const toObject = map => Object.fromEntries(map);
  const nestedToObject = map => Object.fromEntries([...map].map(([key, value]) => [key, toObject(value)]));

  self.postMessage({
    type: "summary",
    total: matchingCount,
    allTotal: totalRecords,
    median,
    cancerCount: cancers.size,
    emirateCount: emirates.size,
    outcomes: [...outcomeCounts.keys()].sort((a, b) => a.localeCompare(b)),
    outcomeCounts: toObject(outcomeCounts),
    stages: nestedToObject(stageCounts),
    years: toObject(yearCounts),
    breakdownField,
    breakdown: nestedToObject(breakdownCounts)
  });
}

self.onmessage = event => {
  if (event.data?.type === "filter" && records.length) summarize(event.data.filters);
};

loadDataset().catch(error => self.postMessage({ type: "error", message: error.message || "Could not load dashboard data." }));
