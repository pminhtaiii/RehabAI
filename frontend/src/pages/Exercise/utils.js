/**
 * Lightweight DataFrame replacement for pandas-js.
 * Supports only the operations used in Exercise.jsx.
 */
export class MiniDataFrame {
  constructor(data) {
    // data: array of row objects [{ col1: val, col2: val }, ...]
    this._data = Array.isArray(data) ? data : [];
  }

  /** Slice rows: iloc([start, end]) — end is exclusive */
  iloc(range) {
    return new MiniDataFrame(this._data.slice(range[0], range[1]));
  }

  /** Get a column; returns { values: { toArray() } } */
  get(colName) {
    const values = this._data.map(row => row[colName] ?? 0);
    return { values: { toArray: () => values } };
  }

  /** Serialize to CSV string */
  to_csv() {
    if (this._data.length === 0) return '';
    const headers = Object.keys(this._data[0]);
    const rows = this._data.map(row =>
      headers.map(h => (row[h] ?? '')).join(',')
    );
    return [headers.join(','), ...rows].join('\n');
  }
}

export function csvToJSON(csv) {
  var lines = csv.split("\n");
  var result = [];
  var headers;

  headers = lines[0].split(",");

  
  // Original references have left and right swapped, so we swap them back here
  // Swap header names for "left_" and "right_"
  for (let j = 0; j < headers.length; j++) {
    if (headers[j].startsWith("left_")) {
      headers[j] = headers[j].replace("left_", "right_");
    } else if (headers[j].startsWith("right_")) {
      headers[j] = headers[j].replace("right_", "left_");
    }
  }

  for (var i = 1; i < lines.length; i++) {
    // Initialize an empty object to hold the current row
    var obj = {};

    if (lines[i] == undefined || lines[i].trim() == "") {
      continue;
    }

    // Split the line into words
    var words = lines[i].split(",");
    // Loop over the words and add them to the object using the corresponding headers as keys
    for (let j = 0; j < words.length; j++) {
      obj[headers[j].trim()] = Number(words[j]);
    }

    result.push(obj);
  }

  return result;
}
