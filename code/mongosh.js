// mongosh.js
// DS 4320 Project 2: The Value Gap
// Use this script to connect to the database and verify collections
// Run with: mongosh "your-connection-string" --file mongosh.js

// Select the database
use("Cluster2");

// Log all collection names
print("=== Collections in Cluster2 ===");
db.getCollectionNames().forEach(name => print(name));

// Count documents in each collection
print("\n=== Document Counts ===");
print("player_totals: " + db.player_totals.countDocuments());
print("player_salaries: " + db.player_salaries.countDocuments());

// Preview one document from each collection
print("\n=== Sample: player_totals ===");
printjson(db.player_totals.findOne());

print("\n=== Sample: player_salaries ===");
printjson(db.player_salaries.findOne());

// Verify top earners
print("\n=== Top 5 Salaries ===");
db.player_salaries.find(
    {},
    { Player: 1, Salary: 1, _id: 0 }
).sort({ Salary: -1 }).limit(5).forEach(printjson);

// Verify high scorers
print("\n=== Top 5 Scorers ===");
db.player_totals.find(
    {},
    { Player: 1, PTS: 1, AST: 1, _id: 0 }
).sort({ PTS: -1 }).limit(5).forEach(printjson);
